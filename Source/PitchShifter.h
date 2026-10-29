```cpp
#pragma once
#include <JuceHeader.h>
#include <array>
#include <cmath>
#include <algorithm>

// Streaming, stereo-linked phase-vocoder pitch shifter for monophonic sources.
// Fixed-size buffers; no allocation or locking in processSample().
// Deliberate latency: fftSize samples. Formant preservation is not claimed here.
class FRiKKTUNEPitchShifter
{
public:
    static constexpr int fftOrder = 11;
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr int hopSize = 256;
    static constexpr int bins = fftSize / 2 + 1;
    static constexpr int ringSize = fftSize * 8;
    static constexpr int maxChannels = 2;

    FRiKKTUNEPitchShifter() : fft(fftOrder)
    {
        for (int i = 0; i < fftSize; ++i)
            window[(size_t)i] = std::sqrt(
                0.5f - 0.5f * std::cos(
                    juce::MathConstants<float>::twoPi * (float)i / (float)fftSize));
        reset();
    }

    void prepare(double sr, int channels) noexcept
    {
        sampleRate = std::isfinite(sr) && sr > 1000.0 ? sr : 44100.0;
        activeChannels = juce::jlimit(1, maxChannels, channels);
        formantSmoothingCoefficient =
            1.0f - std::exp(-1.0f / (float)(sampleRate * 0.020));
        formantGainFrameCoefficient =
            1.0f - std::exp(-((float)hopSize / (float)sampleRate) / 0.035f);
        lpcEnvelopeFrameCoefficient =
            1.0f - std::exp(-((float)hopSize / (float)sampleRate) / 0.045f);
        voicednessFrameCoefficient =
            1.0f - std::exp(-((float)hopSize / (float)sampleRate) / 0.025f);
        reset();
    }

    void reset() noexcept
    {
        for (auto& a : inputRing) a.fill(0.0f);
        for (auto& a : outputRing) a.fill(0.0f);
        for (auto& a : previousPhase) a.fill(0.0f);
        for (auto& a : previousMagnitude) a.fill(0.0f);
        for (auto& a : synthesisPhase) a.fill(0.0f);
        for (auto& a : phaseInitialised) a.fill(false);
        for (auto& a : synthesisInitialised) a.fill(false);
        for (auto& a : formantGainState) a.fill(1.0f);
        for (auto& a : lpcEnvelopeState) a.fill(1.0f);
        for (auto& a : lpcEnvelopeInitialised) a.fill(false);
        voicednessFrameState.fill(0.0f);
        for (auto& a : fftInput)
            a.fill(juce::dsp::Complex<float>{0.0f, 0.0f});
        for (auto& a : fftOutput)
            a.fill(juce::dsp::Complex<float>{0.0f, 0.0f});
        writePosition = 0;
        samplesSeen = 0;
        inputFilled = 0;
        currentRatio = targetRatio = 1.0f;
        formantPreservation = targetFormantPreservation;
    }

    void setPitchRatio(float ratio) noexcept
    {
        if (std::isfinite(ratio))
            targetRatio = std::clamp(ratio, 0.5f, 2.0f);
    }

    void setFormantPreservation(float amount) noexcept
    {
        if (std::isfinite(amount))
            targetFormantPreservation = std::clamp(amount, 0.0f, 1.0f);
    }

    void processSample(const float* input, float* output,
                       int channels, float ratio) noexcept
    {
        if (channels <= 0 || output == nullptr)
            return;

        const int chCount = juce::jlimit(1, activeChannels, channels);
        setPitchRatio(ratio);

        const float semitoneDelta = std::abs(
            12.0f * std::log2(std::max(targetRatio, 1.0e-6f)
            / std::max(currentRatio, 1.0e-6f)));

        const float transitionMs =
            3.0f + 9.0f * std::clamp(semitoneDelta / 6.0f, 0.0f, 1.0f);

        const float adaptiveRatioCoefficient =
            1.0f - std::exp(-1.0f / (float)(sampleRate * transitionMs * 0.001));

        currentRatio += adaptiveRatioCoefficient * (targetRatio - currentRatio);
        formantPreservation += formantSmoothingCoefficient
            * (targetFormantPreservation - formantPreservation);

        for (int ch = 0; ch < chCount; ++ch)
        {
            constexpr float maxSafeInput = 16.0f;
            const float raw =
                input != nullptr && std::isfinite(input[ch]) ? input[ch] : 0.0f;
            const float x = std::clamp(raw, -maxSafeInput, maxSafeInput);

            inputRing[(size_t)ch][(size_t)writePosition] = x;
            output[ch] = outputRing[(size_t)ch][(size_t)writePosition];
            outputRing[(size_t)ch][(size_t)writePosition] = 0.0f;
        }

        for (int ch = chCount; ch < channels; ++ch)
            output[ch] = 0.0f;

        ++samplesSeen;
        inputFilled = std::min(inputFilled + 1, fftSize);

        if (inputFilled == fftSize && (samplesSeen % hopSize) == 0)
            for (int ch = 0; ch < chCount; ++ch)
                processFrame(ch);

        writePosition = (writePosition + 1) % ringSize;
    }

private:
    static float wrapPhase(float x) noexcept
    {
        while (x > juce::MathConstants<float>::pi)
            x -= juce::MathConstants<float>::twoPi;
        while (x < -juce::MathConstants<float>::pi)
            x += juce::MathConstants<float>::twoPi;
        return x;
    }

    static bool estimateLpcEnvelope(
        const std::array<juce::dsp::Complex<float>, fftSize>& frame,
        std::array<float, bins>& result,
        float& confidence) noexcept
    {
        confidence = 0.0f;
        constexpr int order = 16;
        std::array<double, order + 1> autocorrelation{};
        std::array<double, order + 1> coefficients{};
        std::array<double, order + 1> previous{};

        for (int lag = 0; lag <= order; ++lag)
        {
            double sum = 0.0;
            for (int n = lag; n < fftSize; ++n)
                sum += (double)frame[(size_t)n].real()
                     * (double)frame[(size_t)(n - lag)].real();
            autocorrelation[(size_t)lag] = sum;
        }

        const double r0 = autocorrelation[0];
        if (!std::isfinite(r0) || r0 < 1.0e-10)
            return false;

        double error = r0;
        coefficients[0] = 1.0;

        for (int i = 1; i <= order; ++i)
        {
            double residual = autocorrelation[(size_t)i];
            for (int j = 1; j < i; ++j)
                residual += coefficients[(size_t)j]
                          * autocorrelation[(size_t)(i - j)];

            double reflection = -residual / std::max(error, 1.0e-12);
            if (!std::isfinite(reflection))
                return false;

            reflection = std::clamp(reflection, -0.98, 0.98);
            previous = coefficients;

            for (int j = 1; j < i; ++j)
                coefficients[(size_t)j] =
                    previous[(size_t)j]
                    + reflection * previous[(size_t)(i - j)];

            coefficients[(size_t)i] = reflection;
            error *= (1.0 - reflection * reflection);

            if (!std::isfinite(error) || error <= r0 * 1.0e-8)
                return false;
        }

        const double gain = std::sqrt(std::max(error, 1.0e-12));

        for (int k = 0; k < bins; ++k)
        {
            const double omega =
                juce::MathConstants<double>::twoPi * (double)k / (double)fftSize;
            double real = 1.0;
            double imag = 0.0;

            for (int i = 1; i <= order; ++i)
            {
                const double angle = omega * (double)i;
                real += coefficients[(size_t)i] * std::cos(angle);
                imag -= coefficients[(size_t)i] * std::sin(angle);
            }

            const double denominator = std::sqrt(real * real + imag * imag);
            const double value = gain / std::max(denominator, 1.0e-5);

            if (!std::isfinite(value))
                return false;

            result[(size_t)k] =
                (float)std::clamp(value, 1.0e-6, 1.0e6);
        }

        confidence = (float)std::clamp(1.0 - error / r0, 0.0, 1.0);
        return std::isfinite(confidence);
    }

    float estimateVoicedness(
        const std::array<juce::dsp::Complex<float>, fftSize>& frame) const noexcept
    {
        const int minLag = juce::jlimit(
            2, fftSize / 4, (int)(sampleRate / 500.0));
        const int maxLag = juce::jlimit(
            minLag + 1, fftSize / 2, (int)(sampleRate / 70.0));

        double energy = 0.0;
        for (int n = 0; n < fftSize; ++n)
        {
            const double x = frame[(size_t)n].real();
            energy += x * x;
        }

        if (!std::isfinite(energy) || energy < 1.0e-9)
            return 0.0f;

        double best = 0.0;

        for (int lag = minLag; lag <= maxLag; lag += 2)
        {
            double cross = 0.0;
            double e1 = 0.0;
            double e2 = 0.0;

            for (int n = lag; n < fftSize; n += 4)
            {
                const double a = frame[(size_t)n].real();
                const double b = frame[(size_t)(n - lag)].real();
                cross += a * b;
                e1 += a * a;
                e2 += b * b;
            }

            const double denom = std::sqrt(std::max(1.0e-18, e1 * e2));
            if (denom > 1.0e-9)
                best = std::max(best, cross / denom);
        }

        return std::isfinite(best)
            ? (float)std::clamp(best, 0.0, 1.0) : 0.0f;
    }

    void processFrame(int ch) noexcept
    {
        auto& in = fftInput[(size_t)ch];
        auto& out = fftOutput[(size_t)ch];

        in.fill(juce::dsp::Complex<float>{0.0f, 0.0f});

        const int start = (writePosition + ringSize - fftSize + 1) % ringSize;

        for (int i = 0; i < fftSize; ++i)
        {
            const float x = inputRing[(size_t)ch]
                [(size_t)((start + i) % ringSize)] * window[(size_t)i];

            in[(size_t)i].real(x);
            in[(size_t)i].imag(0.0f);
        }

        fft.perform(in.data(), out.data(), false);

        const float measuredVoicedness = estimateVoicedness(in);
        auto& voicednessState = voicednessFrameState[(size_t)ch];
        voicednessState += voicednessFrameCoefficient
            * (measuredVoicedness - voicednessState);

        const float voicedGate = juce::jlimit(
            0.0f, 1.0f, (voicednessState - 0.18f) / 0.42f);

        std::array<float, bins> magnitude{};
        std::array<float, bins> phase{};
        std::array<float, bins> instOmega{};
        std::array<float, bins> envelope{};

        auto& prev = previousPhase[(size_t)ch];
        auto& prevMag = previousMagnitude[(size_t)ch];
        constexpr float twoPi = juce::MathConstants<float>::twoPi;

        for (int k = 0; k < bins; ++k)
        {
            const float re = out[(size_t)k].real();
            const float im = out[(size_t)k].imag();

            magnitude[(size_t)k] = std::sqrt(re * re + im * im);
            phase[(size_t)k] = std::atan2(im, re);

            const float expected =
                twoPi * (float)k * (float)hopSize / (float)fftSize;

            if (!phaseInitialised[(size_t)ch][(size_t)k])
            {
                instOmega[(size_t)k] =
                    twoPi * (float)k / (float)fftSize;
                prev[(size_t)k] = phase[(size_t)k];
                phaseInitialised[(size_t)ch][(size_t)k] = true;
            }
            else
            {
                const float delta =
                    wrapPhase(phase[(size_t)k] - prev[(size_t)k] - expected);

                instOmega[(size_t)k] =
                    twoPi * (float)k / (float)fftSize
                    + delta / (float)hopSize;

                prev[(size_t)k] = phase[(size_t)k];
            }
        }

        double positiveFlux = 0.0;
        double previousEnergy = 0.0;

        for (int k = 0; k < bins; ++k)
        {
            const float oldMag = prevMag[(size_t)k];
            const float newMag = magnitude[(size_t)k];

            positiveFlux += (double)juce::jmax(0.0f, newMag - oldMag);
            previousEnergy += (double)oldMag;
            prevMag[(size_t)k] = newMag;
        }

        const bool transientFrame = previousEnergy > 1.0e-5
            && positiveFlux / previousEnergy > 0.45;

        constexpr double envelopeFloor = 1.0e-9;
        std::array<double, bins + 1> prefixMagnitude{};
        std::array<double, bins + 1> prefixIndexMagnitude{};

        for (int k = 0; k < bins; ++k)
        {
            const double logMag = std::log(std::max(
                envelopeFloor, (double)magnitude[(size_t)k]));

            prefixMagnitude[(size_t)k + 1] =
                prefixMagnitude[(size_t)k] + logMag;

            prefixIndexMagnitude[(size_t)k + 1] =
                prefixIndexMagnitude[(size_t)k] + (double)k * logMag;
        }

        for (int k = 0; k < bins; ++k)
        {
            const int radius = juce::jlimit(2, 24, k / 12);
            const int left = juce::jmax(0, k - radius);
            const int right = juce::jmin(bins - 1, k + radius);

            const double leftSum = prefixMagnitude[(size_t)k + 1]
                                 - prefixMagnitude[(size_t)left];
            const double leftIndexSum = prefixIndexMagnitude[(size_t)k + 1]
                                      - prefixIndexMagnitude[(size_t)left];

            const double rightSum = prefixMagnitude[(size_t)right + 1]
                                  - prefixMagnitude[(size_t)k + 1];
            const double rightIndexSum =
                prefixIndexMagnitude[(size_t)right + 1]
                - prefixIndexMagnitude[(size_t)k + 1];

            const double denom = (double)(radius + 1);

            double weighted =
                ((denom - (double)k) * leftSum + leftIndexSum
                + (denom + (double)k) * rightSum - rightIndexSum) / denom;

            const double leftCount = (double)(k - left + 1);
            const double leftIndices = (double)(left + k) * leftCount * 0.5;
            const double rightCount = (double)(right - k);
            const double rightIndices =
                (double)(k + 1 + right) * rightCount * 0.5;

            double weightSum =
                ((denom - (double)k) * leftCount + leftIndices
                + (denom + (double)k) * rightCount - rightIndices) / denom;

            const double leftExtraCount =
                (double)juce::jmax(0, radius - k);
            const double leftExtraWeight =
                leftExtraCount * (leftExtraCount + 1.0) / (2.0 * denom);

            const double rightExtraCount =
                (double)juce::jmax(0, k + radius - (bins - 1));
            const double rightExtraWeight =
                rightExtraCount * (rightExtraCount + 1.0) / (2.0 * denom);

            weighted += leftExtraWeight * std::log(
                std::max(envelopeFloor, (double)magnitude[0]))
                + rightExtraWeight * std::log(
                std::max(envelopeFloor, (double)magnitude[(size_t)(bins - 1)]));

            weightSum += leftExtraWeight + rightExtraWeight;

            const double logEnvelope = weighted / juce::jmax(1.0e-12, weightSum);
            envelope[(size_t)k] = (float)std::exp(logEnvelope);
        }

        std::array<float, bins> lpcEnvelope{};
        float lpcConfidence = 0.0f;
        auto& smoothedLpc = lpcEnvelopeState[(size_t)ch];

        bool lpcValid = estimateLpcEnvelope(in, lpcEnvelope, lpcConfidence)
            && lpcConfidence > 0.08f;

        if (lpcValid && !transientFrame)
        {
            double frameLogMean = 0.0;

            for (int k = 1; k < bins; ++k)
                frameLogMean += std::log(
                    std::max(1.0e-9f, lpcEnvelope[(size_t)k]));

            frameLogMean /= (double)(bins - 1);

            for (int k = 0; k < bins; ++k)
            {
                const float normalised = (float)std::exp(std::clamp(
                    std::log(std::max(1.0e-9f, lpcEnvelope[(size_t)k]))
                    - frameLogMean, -18.0, 18.0));

                if (!lpcEnvelopeInitialised[(size_t)ch][(size_t)k])
                {
                    smoothedLpc[(size_t)k] = normalised;
                    lpcEnvelopeInitialised[(size_t)ch][(size_t)k] = true;
                }
                else
                {
                    const float oldLog = std::log(
                        std::max(1.0e-9f, smoothedLpc[(size_t)k]));
                    const float newLog = std::log(
                        std::max(1.0e-9f, normalised));

                    smoothedLpc[(size_t)k] = std::exp(
                        oldLog + lpcEnvelopeFrameCoefficient
                        * (newLog - oldLog));
                }
            }
        }

        lpcValid = lpcValid && lpcEnvelopeInitialised[(size_t)ch][1];

        if (lpcValid)
        {
            double smoothLogMean = 0.0;
            double lpcLogMean = 0.0;

            for (int k = 1; k < bins; ++k)
            {
                smoothLogMean += std::log(
                    std::max(1.0e-9f, envelope[(size_t)k]));
                lpcLogMean += std::log(
                    std::max(1.0e-9f, smoothedLpc[(size_t)k]));
            }

            smoothLogMean /= (double)(bins - 1);
            lpcLogMean /= (double)(bins - 1);

            for (int k = 0; k < bins; ++k)
            {
                const double smoothLog =
                    std::log(std::max(1.0e-9f, envelope[(size_t)k]))
                    - smoothLogMean;

                const double lpcLog =
                    std::log(std::max(1.0e-9f, smoothedLpc[(size_t)k]))
                    - lpcLogMean;

                const double lpcWeight = 0.25 * (double)voicedGate
                    * (double)juce::jlimit(
                        0.0f, 1.0f, (lpcConfidence - 0.08f) / 0.42f);

                const double blendedLog =
                    (1.0 - lpcWeight) * smoothLog + lpcWeight * lpcLog;

                envelope[(size_t)k] =
                    (float)std::exp(std::clamp(blendedLog, -18.0, 18.0));
            }
        }

        out.fill(juce::dsp::Complex<float>{0.0f, 0.0f});
        auto& syn = synthesisPhase[(size_t)ch];

        for (int k = 0; k < bins; ++k)
        {
            const float sourcePosition = (float)k / currentRatio;

            if (sourcePosition >= (float)(bins - 1))
            {
                syn[(size_t)k] = 0.0f;
                synthesisInitialised[(size_t)ch][(size_t)k] = false;
                continue;
            }

            const int lo = (int)sourcePosition;
            const int hi = lo + 1;
            const float frac = sourcePosition - (float)lo;

            const float omega = instOmega[(size_t)lo]
                + frac * (instOmega[(size_t)hi] - instOmega[(size_t)lo]);

            if (transientFrame || std::abs(currentRatio - 1.0f) < 1.0e-4f)
            {
                syn[(size_t)k] = phase[(size_t)lo];
                synthesisInitialised[(size_t)ch][(size_t)k] = true;
            }
            else if (!synthesisInitialised[(size_t)ch][(size_t)k])
            {
                syn[(size_t)k] = phase[(size_t)lo];
                synthesisInitialised[(size_t)ch][(size_t)k] = true;
            }
            else
            {
                syn[(size_t)k] += omega * currentRatio * (float)hopSize;
            }
        }

        for (int k = 0; k < bins; ++k)
        {
            const float sourcePosition = (float)k / currentRatio;
            if (sourcePosition >= (float)(bins - 1))
                continue;

            const int lo = (int)sourcePosition;
            const int hi = lo + 1;
            const float frac = sourcePosition - (float)lo;

            const float loMagnitude = magnitude[(size_t)lo];
            const float hiMagnitude = magnitude[(size_t)hi];

            const float sourceMagnitude = std::sqrt(std::max(0.0f,
                (1.0f - frac) * loMagnitude * loMagnitude
                + frac * hiMagnitude * hiMagnitude));

            const float sourceEnvelope = envelope[(size_t)lo]
                + frac * (envelope[(size_t)hi] - envelope[(size_t)lo]);

            const float targetEnvelope = envelope[(size_t)k];
            const float safeSourceEnvelope = std::max(sourceEnvelope, 1.0e-5f);
            const float excitationMagnitude = sourceMagnitude / safeSourceEnvelope;
            const float recolouredMagnitude =
                excitationMagnitude * std::max(targetEnvelope, 1.0e-5f);

            const float formantRatio = std::clamp(
                recolouredMagnitude / std::max(sourceMagnitude, 1.0e-6f),
                0.5f, 2.0f);

            const float desiredFormantGain =
                (!transientFrame && formantPreservation > 0.0f
                 && sourceEnvelope > 1.0e-7f && targetEnvelope > 1.0e-7f)
                ? std::exp(voicedGate * std::log(formantRatio)) : 1.0f;

            auto& gainState = formantGainState[(size_t)ch][(size_t)k];

            const float logCurrent =
                std::log(std::clamp(gainState, 0.5f, 2.0f));
            const float logTarget = std::log(desiredFormantGain);

            gainState = std::clamp(
                std::exp(logCurrent + formantGainFrameCoefficient
                    * (logTarget - logCurrent)), 0.5f, 2.0f);

            const float mag = sourceMagnitude
                * std::pow(gainState, formantPreservation);

            float outputPhase = syn[(size_t)k];

            if (!transientFrame && std::abs(currentRatio - 1.0f) >= 1.0e-4f)
            {
                int peak = lo;
                const int searchLeft = juce::jmax(1, lo - 4);
                const int searchRight = juce::jmin(bins - 2, lo + 4);

                for (int q = searchLeft; q <= searchRight; ++q)
                {
                    if (magnitude[(size_t)q] >= magnitude[(size_t)(q - 1)]
                        && magnitude[(size_t)q] > magnitude[(size_t)(q + 1)]
                        && magnitude[(size_t)q] > magnitude[(size_t)peak])
                    {
                        peak = q;
                    }
                }

                const int targetPeak = juce::jlimit(
                    0, bins - 1, (int)std::round((float)peak * currentRatio));

                const float relativePhase =
                    wrapPhase(phase[(size_t)lo] - phase[(size_t)peak]);

                outputPhase = syn[(size_t)targetPeak] + relativePhase;
            }

            const float re = mag * std::cos(outputPhase);
            const float im = mag * std::sin(outputPhase);

            out[(size_t)k].real(re);
            out[(size_t)k].imag(im);

            if (k > 0 && k < fftSize / 2)
            {
                out[(size_t)(fftSize - k)].real(re);
                out[(size_t)(fftSize - k)].imag(-im);
            }
        }

        out[0].imag(0.0f);
        out[(size_t)(fftSize / 2)].imag(0.0f);

        fft.perform(out.data(), in.data(), true);

        const int outputStart = (writePosition + 1) % ringSize;

        for (int i = 0; i < fftSize; ++i)
        {
            const float y = in[(size_t)i].real()
                * window[(size_t)i] * 0.25f;

            const int idx = (outputStart + i) % ringSize;
            outputRing[(size_t)ch][(size_t)idx] +=
                std::isfinite(y) ? y : 0.0f;
        }
    }

    juce::dsp::FFT fft;
    double sampleRate = 44100.0;
    int activeChannels = 2;
    int writePosition = 0;
    int samplesSeen = 0;
    int inputFilled = 0;

    float currentRatio = 1.0f;
    float targetRatio = 1.0f;
    float formantPreservation = 0.75f;
    float targetFormantPreservation = 0.75f;
    float formantSmoothingCoefficient = 0.001f;
    float formantGainFrameCoefficient = 0.2f;
    float lpcEnvelopeFrameCoefficient = 0.05f;
    float voicednessFrameCoefficient = 0.2f;

    std::array<float, maxChannels> voicednessFrameState{};
    std::array<float, fftSize> window{};
    std::array<std::array<float, ringSize>, maxChannels> inputRing{};
    std::array<std::array<float, ringSize>, maxChannels> outputRing{};

    std::array<std::array<juce::dsp::Complex<float>, fftSize>, maxChannels> fftInput{};
    std::array<std::array<juce::dsp::Complex<float>, fftSize>, maxChannels> fftOutput{};

    std::array<std::array<float, bins>, maxChannels> previousPhase{};
    std::array<std::array<float, bins>, maxChannels> previousMagnitude{};
    std::array<std::array<float, bins>, maxChannels> synthesisPhase{};

    std::array<std::array<bool, bins>, maxChannels> phaseInitialised{};
    std::array<std::array<bool, bins>, maxChannels> synthesisInitialised{};

    std::array<std::array<float, bins>, maxChannels> formantGainState{};
    std::array<std::array<float, bins>, maxChannels> lpcEnvelopeState{};
    std::array<std::array<bool, bins>, maxChannels> lpcEnvelopeInitialised{};
};
```

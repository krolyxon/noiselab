function y = spectral_subtraction_reference(x, fs, cfg)
%SPECTRAL_SUBTRACTION_REFERENCE  Plain-MATLAB version of the basic Simulink model.
%
%   y = spectral_subtraction_reference(x, fs)          % defaults from nl_default_config()
%   y = spectral_subtraction_reference(x, fs, cfg)
%
%   Mirrors the Simulink block chain step by step, with no toolbox needed:
%   a frame loop with a buffer, Hann window, FFT, magnitude subtraction with
%   floor, noisy phase, IFFT and overlap-add with a one-frame delay. Uses:
%     * debugging: if Simulink and Python disagree, compare each against this;
%     * the viva: every line maps to one block in the model.
%   The output is latency-compensated and trimmed to length(x), like run_and_export.
%
%   *** UNTESTED: written without access to MATLAB. Run and fix in MATLAB. ***
%   (The same chain is emulated in Python by tests/test_simulink_chain.py, which
%    *is* tested and matches the Python STFT pipeline to within 1e-10.)

if nargin < 3, cfg = nl_default_config(); end
x = mean(x, 2);                                % mono
N = cfg.N;  H = round(N * (1 - cfg.overlap));
w = nl_hann_periodic(N);                       % Hann window block (Constant)
noiseMag = nl_compute_noise_mag(x, N, H, cfg.noiseSeconds, fs);   % InitFcn, Eq. (4)

nSteps = ceil(numel(x) / H) + 1;
xin = [x; zeros(nSteps * H - numel(x), 1)];    % source outputs zeros after the end
buf = zeros(N, 1);                             % Buffer block state (initial conditions 0)
prevSecond = zeros(H, 1);                      % Unit Delay state
out = zeros(nSteps * H, 1);

for k = 1:nSteps
    hop = xin((k-1)*H + (1:H));                % source block: H new samples
    buf = [buf(H+1:end); hop];                 % Buffer: N samples, overlap N - H
    Y = fft(buf .* w);                         % Product + FFT          Eq. (2)
    mag = abs(Y);  ph = angle(Y);              % Complex to Mag-Angle   Eq. (3)
    S = max(mag - cfg.alpha * noiseMag, cfg.beta * noiseMag);   % Subtract, MinMax  Eq. (7)
    frame = real(ifft(S .* exp(1j * ph)));     % Mag-Angle to Complex, IFFT, Re   Eq. (12)
    out((k-1)*H + (1:H)) = frame(1:H) + prevSecond;   % Selector x2, Unit Delay, Add  Eq. (13)
    prevSecond = frame(H+1:N);
end

y = out(H+1 : H+numel(x));                     % remove one hop of latency, trim
pk = max(abs(y));
if pk > 0.99, y = y * (0.99 / pk); end         % Eq. (14)
end

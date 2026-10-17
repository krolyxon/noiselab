function noiseMag = nl_compute_noise_mag(x, N, H, noiseSeconds, fs)
%NL_COMPUTE_NOISE_MAG  Noise magnitude estimate |N^_k| from the first T seconds, Eq. (4).
%
%   noiseMag = nl_compute_noise_mag(x, N, H, noiseSeconds, fs)
%
%   x            mono signal (column vector)
%   N, H         frame length and hop (samples)
%   noiseSeconds T, the length of the noise-only lead-in
%   fs           sample rate (Hz)
%
%   Returns an N-by-1 vector (full spectrum, bins 0..N-1) = mean |FFT| over the
%   complete frames that start at samples 0, H, 2H, ... and end at or before
%   T*fs. If T is shorter than one frame, the single frame starting at 0 is used.
%   This is exactly the frame set used by the Python code (estimate_initial).
%
%   *** UNTESTED: written without access to MATLAB. Run and fix in MATLAB. ***

x = x(:);
w = nl_hann_periodic(N);
lastSample = max(round(noiseSeconds * fs), N);      % frames must end by here
lastSample = min(lastSample, numel(x));
starts = 0:H:(lastSample - N);                      % 0-based frame starts
if isempty(starts)
    error('nl_compute_noise_mag:tooShort', ...
        'Signal is shorter than one frame (%d samples).', N);
end

mags = zeros(N, numel(starts));
for i = 1:numel(starts)
    frame = x(starts(i) + (1:N)) .* w;              % Eq. (1)-(2): frame and window
    mags(:, i) = abs(fft(frame));                   % Eq. (3): magnitude
end
noiseMag = mean(mags, 2);                           % Eq. (4): average over frames
end

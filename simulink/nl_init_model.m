function nl_init_model(cfg)
%NL_INIT_MODEL  Model InitFcn: load the input file and set all model variables.
%
%   Called from the model's InitFcn callback (set by build_spectral_subtraction_model.m)
%   with the config stored in the base-workspace variable nl_cfg. Creates these
%   base-workspace variables, which the blocks reference by name:
%
%     nl_N, nl_H        frame length and hop
%     nl_fs             sample rate of the input file
%     nl_x              mono input signal (used by the Signal From Workspace source)
%     nl_hannWin        periodic Hann window (N-by-1)
%     noiseMag          noise magnitude estimate (N-by-1), Eq. (4)
%     nl_alpha, nl_beta subtraction parameters, Eq. (7)
%     nl_stopTime       simulation stop time covering the whole file plus one hop
%
%   *** UNTESTED: written without access to MATLAB. Run and fix in MATLAB. ***

if nargin < 1
    cfg = evalin('base', 'nl_cfg');
end

[x, fs] = audioread(cfg.inputFile);
x = mean(x, 2);                          % mono downmix (the model is single-channel)

N = cfg.N;
H = round(N * (1 - cfg.overlap));
if H ~= N/2
    warning('nl_init_model:overlap', ...
        ['The basic model''s overlap-add (split frame in halves + Unit Delay) assumes ' ...
         '50 %% overlap. Use overlap = 0.5.']);
end

noiseMag = nl_compute_noise_mag(x, N, H, cfg.noiseSeconds, fs);

% Number of hop-sized steps: enough to output every input sample. The
% overlap-add delays the output by one hop (H samples), so run one extra step.
nSteps = ceil(numel(x) / H) + 1;

assignin('base', 'nl_N', N);
assignin('base', 'nl_H', H);
assignin('base', 'nl_fs', fs);
assignin('base', 'nl_x', x);
assignin('base', 'nl_hannWin', nl_hann_periodic(N));
assignin('base', 'noiseMag', noiseMag);
assignin('base', 'nl_alpha', cfg.alpha);
assignin('base', 'nl_beta', cfg.beta);
assignin('base', 'nl_stopTime', (nSteps - 1) * H / fs);
assignin('base', 'nl_nInput', numel(x));
end

function y = run_and_export(noisyWav, outWav, cfg)
%RUN_AND_EXPORT  Run the basic Simulink model on a WAV file and write the cleaned WAV.
%
%   run_and_export('samples/noisy.wav')                       % -> cleaned_simulink.wav
%   run_and_export('samples/noisy.wav', 'out/cleaned_simulink.wav')
%   y = run_and_export(noisyWav, outWav, cfg)                  % custom nl_default_config()
%
%   Steps
%     1. Make a mono copy of the input if it is stereo (the model is single-channel).
%     2. Build the model if simulink/<modelName>.slx does not exist yet, else load it.
%     3. Point the source at this file, disable the audio-device block (batch run).
%     4. sim() the model, read the cleaned signal from the To Workspace block.
%     5. Remove the one-hop (H-sample) latency of the overlap-add, trim to the
%        input length, apply the same peak protection as Python (Eq. 14), and write
%        a 16-bit WAV at the input sample rate.
%
%   The written file is what scripts/compare_simulink_python.py expects. (The
%   comparison script also aligns by cross-correlation, so a residual latency from
%   the Buffer block in your release does not break the comparison; it is reported.)
%
%   *** UNTESTED: written without access to MATLAB. Run and fix in MATLAB. ***

here = fileparts(mfilename('fullpath'));
addpath(here);
if nargin < 3 || isempty(cfg), cfg = nl_default_config(); end
if nargin >= 1 && ~isempty(noisyWav), cfg.inputFile = noisyWav; end
if nargin >= 2 && ~isempty(outWav),   cfg.outputFile = outWav;  end

% ---- 1. mono input ---------------------------------------------------------------
[x, fs] = audioread(cfg.inputFile);
if size(x, 2) > 1
    mono = fullfile(tempdir, 'nl_noisy_mono.wav');
    audiowrite(mono, mean(x, 2), fs);
    fprintf('Input is %d-channel; using a mono downmix: %s\n', size(x, 2), mono);
    cfg.inputFile = mono;
end
assignin('base', 'nl_cfg', cfg);

% ---- 2. build or load ------------------------------------------------------------
mdl = cfg.modelName;
slx = fullfile(here, [mdl '.slx']);
if ~bdIsLoaded(mdl)
    if exist(slx, 'file')
        load_system(slx);
    else
        build_spectral_subtraction_model(cfg);
    end
end

% ---- 3. per-run settings ---------------------------------------------------------
src = [mdl '/Noisy source'];
try
    % From Multimedia File: point it at this run's file. (Signal From Workspace has
    % no such parameter; it reads nl_x, which the InitFcn loads from cfg.inputFile.)
    set_param(src, 'inputFilename', cfg.inputFile);
catch
end
spk = [mdl '/Speaker'];
if getSimulinkBlockHandle(spk) ~= -1
    set_param(spk, 'Commented', 'on');    % no sound card needed for a batch run
end

% ---- 4. simulate -----------------------------------------------------------------
nl_init_model(cfg);                       % also run by the InitFcn; here for nl_stopTime
stopTime = evalin('base', 'nl_stopTime');
H = evalin('base', 'nl_H');
n = evalin('base', 'nl_nInput');
fprintf('Simulating %s: %d samples @ %d Hz, stop time %.3f s ...\n', mdl, n, fs, stopTime);
simOut = sim(mdl, 'StopTime', sprintf('%.12g', stopTime), 'ReturnWorkspaceOutputs', 'on');

if getSimulinkBlockHandle(spk) ~= -1
    set_param(spk, 'Commented', 'off');
end

% ---- 5. collect, align, write ----------------------------------------------------
frames = simOut.get('nl_cleanedFrames');
if ndims(frames) == 3                     % H x 1 x K  (3-D array save format)
    frames = reshape(frames, H, []);
end
y = double(frames(:));                    % column-major = frames in time order

% Overlap-add output at step k holds samples [(k-1)H, kH): one hop of latency.
y = y(H+1:end);
if numel(y) < n
    y(end+1:n) = 0;
end
y = y(1:n);

pk = max(abs(y));
if pk > 0.99                              % Eq. (14): scale down only if needed
    y = y * (0.99 / pk);
end
audiowrite(cfg.outputFile, y, fs, 'BitsPerSample', 16);
fprintf('Wrote %s (%d samples @ %d Hz).\n', cfg.outputFile, numel(y), fs);
fprintf('Compare with Python:\n  python scripts/compare_simulink_python.py "%s" "%s"\n', ...
    cfg.inputFile, cfg.outputFile);
end

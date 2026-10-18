function nl_run_all(noisyWav)
%NL_RUN_ALL  One entry point: run everything in this folder on one WAV file.
%
%   nl_run_all                    % uses noisy.wav in this folder (or ../samples/noisy.wav)
%   nl_run_all('my_noisy.wav')
%
%   Works in MATLAB Online: upload this folder plus a noisy WAV, cd into the
%   folder, and type  nl_run_all . Steps:
%     0. Check that Simulink and DSP System Toolbox are licensed.
%     1. Plain-MATLAB reference (no Simulink needed)    -> cleaned_reference.wav
%     2. Build the Simulink model (no audio device,
%        since MATLAB Online has no sound card)          -> spectral_subtraction_basic.slx
%     3. Simulate it and export the output               -> cleaned_simulink.wav
%     4. Compare 1 vs 3 in MATLAB: SNR between the two outputs.
%   Each step is reported separately, so if step 2 or 3 fails, step 1's result
%   is still written and the error message says which block or parameter broke.
%
%   Then download cleaned_simulink.wav and run on your own machine:
%     python scripts/compare_simulink_python.py noisy.wav cleaned_simulink.wav
%
%   *** UNTESTED: written without access to MATLAB. Run and fix in MATLAB. ***

here = fileparts(mfilename('fullpath'));
addpath(here);

% ---- input file -------------------------------------------------------------------
if nargin < 1 || isempty(noisyWav)
    candidates = {fullfile(here, 'noisy.wav'), fullfile(here, '..', 'samples', 'noisy.wav')};
    noisyWav = '';
    for i = 1:numel(candidates)
        if exist(candidates{i}, 'file'), noisyWav = candidates{i}; break; end
    end
    if isempty(noisyWav)
        error('nl_run_all:noInput', ['No input WAV found. Upload a noisy WAV as %s, ' ...
            'or call nl_run_all(''path/to/file.wav'').'], candidates{1});
    end
end
[x, fs] = audioread(noisyWav);
fprintf('Input: %s  (%d samples, %d Hz, %d channel(s), %.2f s)\n', ...
    noisyWav, size(x, 1), fs, size(x, 2), size(x, 1) / fs);

% ---- 0. toolboxes -----------------------------------------------------------------
hasSimulink = license('test', 'Simulink') && ~isempty(ver('simulink'));
hasDsp = license('test', 'Signal_Blocks') && ~isempty(ver('dsp'));
fprintf('Simulink: %s   DSP System Toolbox: %s\n', yesno(hasSimulink), yesno(hasDsp));

cfg = nl_default_config();
cfg.inputFile = noisyWav;
cfg.addAudioDevice = false;          % no sound card in MATLAB Online / batch runs

% ---- 1. plain-MATLAB reference ----------------------------------------------------
fprintf('\n[1/4] Plain-MATLAB reference loop ...\n');
yRef = spectral_subtraction_reference(x, fs, cfg);
refWav = fullfile(here, 'cleaned_reference.wav');
audiowrite(refWav, yRef, fs, 'BitsPerSample', 16);
fprintf('      wrote %s\n', refWav);

if ~(hasSimulink && hasDsp)
    fprintf(['\nStopping here: Simulink and DSP System Toolbox are both needed for steps 2-4.\n' ...
        'Check your licence (type  ver ). The reference output above still works for the comparison script.\n']);
    return
end

% ---- 2. build ---------------------------------------------------------------------
fprintf('\n[2/4] Building the Simulink model ...\n');
slx = fullfile(here, [cfg.modelName '.slx']);
if bdIsLoaded(cfg.modelName), close_system(cfg.modelName, 0); end
if exist(slx, 'file')
    fprintf('      %s already exists: using it (delete it to rebuild).\n', slx);
else
    try
        build_spectral_subtraction_model(cfg);
    catch err
        fprintf(2, '      BUILD FAILED: %s\n', err.message);
        fprintf(2, '      Fix the line it names in build_spectral_subtraction_model.m, or build by hand (README §3).\n');
        rethrow(err);
    end
end

% ---- 3. simulate + export ---------------------------------------------------------
fprintf('\n[3/4] Simulating and exporting ...\n');
simWav = fullfile(here, 'cleaned_simulink.wav');
try
    ySim = run_and_export(noisyWav, simWav, cfg);
catch err
    fprintf(2, '      SIMULATION FAILED: %s\n', err.message);
    rethrow(err);
end

% ---- 4. Simulink vs reference -----------------------------------------------------
fprintf('\n[4/4] Simulink vs plain-MATLAB reference\n');
n = min(numel(ySim), numel(yRef));
d = ySim(1:n) - yRef(1:n);
snrBetween = 10 * log10(sum(yRef(1:n).^2) / max(sum(d.^2), eps));
fprintf('      SNR between outputs: %.1f dB  (> 40 dB = MATCH; 16-bit WAV rounding limits it to ~90 dB)\n', snrBetween);
if snrBetween < 40
    [c, lags] = xcorr(ySim(1:n), yRef(1:n), 4 * cfg.N);
    [~, i] = max(abs(c));
    fprintf('      Best alignment lag: %d samples (0 expected). A lag that is a multiple of %d\n', ...
        lags(i), round(cfg.N * (1 - cfg.overlap)));
    fprintf('      means the Buffer block adds extra delay in your release; the Python script aligns for it.\n');
end
fprintf('\nDone. Download cleaned_simulink.wav and run:\n  python scripts/compare_simulink_python.py noisy.wav cleaned_simulink.wav\n');
end

function s = yesno(b)
if b, s = 'yes'; else, s = 'NO'; end
end

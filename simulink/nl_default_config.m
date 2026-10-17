function cfg = nl_default_config()
%NL_DEFAULT_CONFIG  Parameters of the basic spectral-subtraction Simulink model.
%
%   *** UNTESTED: written without access to MATLAB. Run and fix in MATLAB. ***
%
%   These values are identical to the "Match Simulink basic" preset of the
%   NoiseLab web app (app/params.py). If you change one here, change it there
%   too, otherwise scripts/compare_simulink_python.py will report differences.
%
%   Equation numbers refer to docs/ALGORITHM.md.

cfg.modelName    = 'spectral_subtraction_basic';
cfg.inputFile    = fullfile(fileparts(mfilename('fullpath')), '..', 'samples', 'noisy.wav');
cfg.outputFile   = 'cleaned_simulink.wav';

cfg.N            = 512;    % frame length (FFT size)                       Eq. (1)
cfg.overlap      = 0.5;    % 50 % overlap -> hop H = N/2 = 256             Eq. (1)
cfg.noiseSeconds = 0.5;    % first T seconds assumed noise-only            Eq. (4)
cfg.alpha        = 2.0;    % over-subtraction factor                       Eq. (7)
cfg.beta         = 0.02;   % spectral floor                                Eq. (7)
% The basic model uses magnitude subtraction (exponent p = 1):
%   |S~| = max(|Y| - alpha*|N^|, beta*|N^|)

% Source block: 'multimedia' = From Multimedia File (as in the block table),
%               'workspace'  = Signal From Workspace (fallback, same result).
cfg.source       = 'multimedia';
cfg.addAudioDevice     = true;   % To Audio Device / Audio Device Writer (listen)
cfg.addSpectrumAnalyzer = true;  % noisy vs cleaned spectrum / spectrogram
end

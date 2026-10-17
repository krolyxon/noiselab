function mdl = build_spectral_subtraction_model(cfg)
%BUILD_SPECTRAL_SUBTRACTION_MODEL  Programmatically build the basic spectral-subtraction model.
%
%   build_spectral_subtraction_model            % uses nl_default_config()
%   build_spectral_subtraction_model(cfg)       % custom config struct
%
%   Creates and saves simulink/<cfg.modelName>.slx with the block chain from
%   simulink/README.md (block table). Requires Simulink + DSP System Toolbox.
%
%   ======================================================================
%   *** UNTESTED: this script was written without access to MATLAB. ***
%   *** Run it, read every warning, and fix block paths/parameter    ***
%   *** names for your release. You may instead build the model by   ***
%   *** hand from the block table in simulink/README.md.             ***
%   ======================================================================
%
%   Defensive design:
%     * addBlock() tries several library paths per block (they differ
%       between releases) and reports all of them if none works.
%     * setParams() never aborts: if a parameter name is wrong for your
%       release it prints a warning listing the block's real dialog
%       parameter names, so you can correct the name here or in the dialog.
%
%   Signal flow (N = 512, H = 256, 50 % overlap; equations in docs/ALGORITHM.md):
%
%   source(H) -> Buffer(N, overlap N-H) -> x hann -> FFT -> |.|,angle
%      |Y| - alpha*noiseMag -> max(., beta*noiseMag)            Eq. (7), p = 1
%      -> Magnitude-Angle to Complex (noisy phase)               Eq. (12)
%      -> IFFT -> Re -> [first half] + UnitDelay([second half])  Eq. (13), OLA
%      -> To Multimedia File / To Workspace / audio device

if nargin < 1 || isempty(cfg)
    cfg = nl_default_config();
end
here = fileparts(mfilename('fullpath'));
addpath(here);                                    % helper functions + InitFcn

mdl = cfg.modelName;
if bdIsLoaded(mdl)
    close_system(mdl, 0);                         % discard any open copy
end
new_system(mdl);
open_system(mdl);

% ---- workspace variables (also re-created by the InitFcn on every run) -------
assignin('base', 'nl_cfg', cfg);
nl_init_model(cfg);                               % defines nl_N, nl_H, noiseMag, ...

% ---- model configuration -----------------------------------------------------
% Fixed-step discrete solver: the model is purely discrete (frame-based DSP).
setParams(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', ...
    'StopTime', 'nl_stopTime');
% PreLoadFcn/InitFcn: make the model self-contained when re-opened later.
% They run in the base workspace and need this folder on the MATLAB path.
setParams(mdl, ...
    'PreLoadFcn', 'if ~exist(''nl_cfg'',''var''), nl_cfg = nl_default_config(); end', ...
    'InitFcn',    'if ~exist(''nl_cfg'',''var''), nl_cfg = nl_default_config(); end; nl_init_model(nl_cfg);');

% ---- layout helper ------------------------------------------------------------
W = 90; Hh = 44;                                  % default block size
pos = @(col, row) [40 + col*140, 40 + row*90, 40 + col*140 + W, 40 + row*90 + Hh];

% ===============================================================================
% 1. Input: From Multimedia File (or Signal From Workspace fallback), H samples/frame
% ===============================================================================
src = [mdl '/Noisy source'];
useWorkspace = strcmp(cfg.source, 'workspace');
if ~useWorkspace
    try
        % DSP System Toolbox > Sources > From Multimedia File
        addBlock(src, {'dspsrcs4/From Multimedia File', 'vipsrcs/From Multimedia File'}, pos(0, 1));
        setParams(src, 'inputFilename', cfg.inputFile);
        % "Samples per audio channel" — the name differs across releases:
        setFirstParam(src, {'audioFrameSize', 'samplesPerAudioFrame', 'SamplesPerAudioFrame'}, 'nl_H');
        setFirstParam(src, {'outputStreams'}, 'Audio only');
        setFirstParam(src, {'audioOutputDataType', 'AudioOutputDataType'}, 'double');
        setFirstParam(src, {'loop', 'PlayCount'}, '1');   % play once (if supported)
    catch err
        warning('nl:source', ['From Multimedia File could not be added/configured (%s).\n' ...
            'Falling back to Signal From Workspace (same signal, read by the InitFcn).'], err.message);
        if getSimulinkBlockHandle(src) ~= -1, delete_block(src); end
        useWorkspace = true;
    end
end
if useWorkspace
    % DSP System Toolbox > Sources > Signal From Workspace
    addBlock(src, {'dspsrcs4/Signal From Workspace'}, pos(0, 1));
    setParams(src, 'X', 'nl_x', 'Ts', '1/nl_fs', 'nsamps', 'nl_H', ...
        'OutputAfterFinalValue', 'Setting to zero');
end

% ===============================================================================
% 2. Framing: Buffer to N samples with overlap N - H                    Eq. (1)
% ===============================================================================
buf = [mdl '/Buffer'];
% DSP System Toolbox > Signal Management > Buffers > Buffer
addBlock(buf, {'dspbuff3/Buffer'}, pos(1, 1));
setParams(buf, 'N', 'nl_N', 'V', 'nl_N - nl_H', 'ic', '0');

% ===============================================================================
% 3. Window: element-wise product with a constant periodic Hann vector  Eq. (2)
%    (Constant + Product instead of the Window Function block, so there is no
%    ambiguity about periodic vs symmetric Hann.)
% ===============================================================================
win = [mdl '/Hann window'];
addBlock(win, {'simulink/Sources/Constant'}, pos(1, 0));
setParams(win, 'Value', 'nl_hannWin', 'VectorParams1D', 'off', 'SampleTime', 'inf');
prodW = [mdl '/Apply window'];
addBlock(prodW, {'simulink/Math Operations/Product'}, pos(2, 1));
setParams(prodW, 'Inputs', '2', 'Multiplication', 'Element-wise(.*)');

% ===============================================================================
% 4. FFT and magnitude / phase                                          Eq. (2)-(3)
% ===============================================================================
fftB = [mdl '/FFT'];
% DSP System Toolbox > Transforms > FFT
addBlock(fftB, {'dspxfrm3/FFT'}, pos(3, 1));
magAng = [mdl '/Mag-Angle'];
addBlock(magAng, {'simulink/Math Operations/Complex to Magnitude-Angle'}, pos(4, 1));
setParams(magAng, 'Output', 'Magnitude and angle');

% ===============================================================================
% 5. Noise estimate (Constant, computed in the InitFcn) and subtraction  Eq. (4), (7)
%    |S~| = max(|Y| - alpha*noiseMag, beta*noiseMag)      (p = 1, magnitude)
% ===============================================================================
nz = [mdl '/noiseMag'];
addBlock(nz, {'simulink/Sources/Constant'}, pos(4, 3));
setParams(nz, 'Value', 'noiseMag', 'VectorParams1D', 'off', 'SampleTime', 'inf');

gA = [mdl '/alpha'];
addBlock(gA, {'simulink/Math Operations/Gain'}, pos(5, 2));
setParams(gA, 'Gain', 'nl_alpha');
gB = [mdl '/beta'];
addBlock(gB, {'simulink/Math Operations/Gain'}, pos(5, 3));
setParams(gB, 'Gain', 'nl_beta');

sub = [mdl '/Subtract'];
try
    addBlock(sub, {'simulink/Math Operations/Subtract'}, pos(6, 1));
catch
    addBlock(sub, {'simulink/Math Operations/Sum'}, pos(6, 1));
end
setParams(sub, 'Inputs', '+-');

mx = [mdl '/Spectral floor (max)'];
addBlock(mx, {'simulink/Math Operations/MinMax'}, pos(7, 1));
setParams(mx, 'Function', 'max', 'Inputs', '2');

% ===============================================================================
% 6. Recombine with the ORIGINAL (noisy) phase                          Eq. (12)
% ===============================================================================
toCplx = [mdl '/Mag-Angle to Complex'];
addBlock(toCplx, {'simulink/Math Operations/Magnitude-Angle to Complex'}, pos(8, 1));
setParams(toCplx, 'Input', 'Magnitude and angle');

% ===============================================================================
% 7. IFFT, then take the real part (the spectrum is conjugate-symmetric, so the
%    imaginary part is only rounding error; taking Re is release-independent)
% ===============================================================================
ifftB = [mdl '/IFFT'];
% DSP System Toolbox > Transforms > IFFT
addBlock(ifftB, {'dspxfrm3/IFFT'}, pos(9, 1));
re = [mdl '/Real part'];
addBlock(re, {'simulink/Math Operations/Complex to Real-Imag'}, pos(10, 1));
setParams(re, 'Output', 'Real');

% ===============================================================================
% 8. Overlap-add for 50 % overlap                                       Eq. (13)
%    out(k) = first half of frame k + second half of frame k-1
%    (periodic Hann at 50 % sums to 1, so no normalisation is needed)
% ===============================================================================
s1 = [mdl '/First half'];
addBlock(s1, {'simulink/Signal Routing/Selector'}, pos(11, 0));
setSelector(s1, '1:nl_H');
s2 = [mdl '/Second half'];
addBlock(s2, {'simulink/Signal Routing/Selector'}, pos(11, 2));
setSelector(s2, '(nl_H+1):nl_N');

dly = [mdl '/One-frame delay'];
addBlock(dly, {'simulink/Discrete/Unit Delay'}, pos(12, 2));
setParams(dly, 'InitialCondition', 'zeros(nl_H,1)');

ola = [mdl '/Overlap-add'];
addBlock(ola, {'simulink/Math Operations/Add'}, pos(13, 1));
setParams(ola, 'Inputs', '++');

% ===============================================================================
% 9. Outputs
% ===============================================================================
toFile = [mdl '/To Multimedia File'];
% DSP System Toolbox > Sinks > To Multimedia File
addBlock(toFile, {'dspsnks4/To Multimedia File', 'vipsnks/To Multimedia File'}, pos(14, 0));
setFirstParam(toFile, {'outputFilename', 'OutputFilename'}, cfg.outputFile);
setFirstParam(toFile, {'streamSelection'}, 'Audio only');
setFirstParam(toFile, {'fileType', 'FileFormat'}, 'WAV');

toWs = [mdl '/To Workspace'];
addBlock(toWs, {'simulink/Sinks/To Workspace'}, pos(14, 1));
setParams(toWs, 'VariableName', 'nl_cleanedFrames', 'SaveFormat', 'Array', 'MaxDataPoints', 'inf');
setFirstParam(toWs, {'Save2DSignal'}, '2-D array (concatenate along first dimension)');

if cfg.addAudioDevice
    spk = [mdl '/Speaker'];
    try
        % Newer releases: Audio Device Writer; older: To Audio Device.
        addBlock(spk, {'dspsnks4/Audio Device Writer', 'audiosinks/Audio Device Writer', ...
                       'dspsnks4/To Audio Device'}, pos(14, 2));
    catch err
        warning('nl:audio', 'No audio-device block found (%s); skipping it.', err.message);
        cfg.addAudioDevice = false;
    end
end

if cfg.addSpectrumAnalyzer
    sa = [mdl '/Spectrum (noisy vs cleaned)'];
    try
        % DSP System Toolbox > Sinks > Spectrum Analyzer (switch View to Spectrogram if wanted)
        addBlock(sa, {'dspsnks4/Spectrum Analyzer'}, pos(14, 3));
        setFirstParam(sa, {'NumInputPorts'}, '2');
    catch err
        warning('nl:sa', 'Spectrum Analyzer not added (%s).', err.message);
        cfg.addSpectrumAnalyzer = false;
    end
end

% ===============================================================================
% 10. Wiring
% ===============================================================================
L = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');
L('Noisy source/1', 'Buffer/1');
L('Buffer/1', 'Apply window/1');
L('Hann window/1', 'Apply window/2');
L('Apply window/1', 'FFT/1');
L('FFT/1', 'Mag-Angle/1');
L('Mag-Angle/1', 'Subtract/1');                       % |Y|
L('noiseMag/1', 'alpha/1');
L('noiseMag/1', 'beta/1');
L('alpha/1', 'Subtract/2');                           % alpha*|N^|
L('Subtract/1', 'Spectral floor (max)/1');
L('beta/1', 'Spectral floor (max)/2');                % beta*|N^|
L('Spectral floor (max)/1', 'Mag-Angle to Complex/1');
L('Mag-Angle/2', 'Mag-Angle to Complex/2');           % noisy phase
L('Mag-Angle to Complex/1', 'IFFT/1');
L('IFFT/1', 'Real part/1');
L('Real part/1', 'First half/1');
L('Real part/1', 'Second half/1');
L('Second half/1', 'One-frame delay/1');
L('First half/1', 'Overlap-add/1');
L('One-frame delay/1', 'Overlap-add/2');
L('Overlap-add/1', 'To Multimedia File/1');
L('Overlap-add/1', 'To Workspace/1');
if cfg.addAudioDevice,      L('Overlap-add/1', 'Speaker/1'); end
if cfg.addSpectrumAnalyzer
    L('Noisy source/1', 'Spectrum (noisy vs cleaned)/1');
    L('Overlap-add/1', 'Spectrum (noisy vs cleaned)/2');
end

% ---- a visible note in the diagram ----------------------------------------------
try
    a = Simulink.Annotation([mdl '/Spectral subtraction (Boll 1979), basic model. ' ...
        'Built by build_spectral_subtraction_model.m (UNTESTED script: verify every block). ' ...
        'Parameters: N=512, 50% Hann, T=0.5 s, alpha=2, beta=0.02, magnitude subtraction.']);
    a.Position = [40 360 900 400];
catch
    % annotations are cosmetic; ignore if the API differs
end

save_system(mdl, fullfile(here, [mdl '.slx']));
fprintf('Saved %s\n', fullfile(here, [mdl '.slx']));
fprintf('Source block: %s\n', ternary(useWorkspace, 'Signal From Workspace', 'From Multimedia File'));
fprintf('Next: run_and_export(''samples/noisy.wav'') or press Run in the model.\n');
end

% =================================================================================
% helpers
% =================================================================================
function addBlock(dest, candidates, position)
%ADDBLOCK  Try library paths in order; error listing all of them if none exists.
lastMsg = '';
for i = 1:numel(candidates)
    try
        add_block(candidates{i}, dest, 'Position', position);
        return
    catch err
        lastMsg = err.message;
    end
end
error('nl:addBlock', 'Could not add "%s". Tried: %s.\nLast error: %s\n%s', dest, ...
    strjoin(candidates, ' | '), lastMsg, ...
    'Find the block in the Library Browser, right-click > Help, and put its path here.');
end

function setParams(blk, varargin)
%SETPARAMS  set_param for name/value pairs; warn (with the valid names) instead of failing.
for k = 1:2:numel(varargin)
    try
        set_param(blk, varargin{k}, varargin{k+1});
    catch err
        warning('nl:param', 'Could not set %s = %s on "%s": %s\nValid dialog parameters: %s', ...
            varargin{k}, char(string(varargin{k+1})), blk, err.message, dialogNames(blk));
    end
end
end

function setFirstParam(blk, names, value)
%SETFIRSTPARAM  Set the first parameter name (from a list of alternatives) that exists.
for i = 1:numel(names)
    try
        set_param(blk, names{i}, value);
        return
    catch
    end
end
warning('nl:param', 'None of {%s} could be set on "%s".\nValid dialog parameters: %s', ...
    strjoin(names, ', '), blk, dialogNames(blk));
end

function setSelector(blk, idx)
%SETSELECTOR  1-D Selector with a dialog index vector (two API generations).
try
    set_param(blk, 'NumberOfDimensions', '1', 'IndexMode', 'One-based', ...
        'IndexOptionArray', {'Index vector (dialog)'}, 'IndexParamArray', {idx}, ...
        'InputPortWidth', 'nl_N');
catch
    setParams(blk, 'IndexMode', 'One-based', 'IndexOptions', 'Index vector (dialog)', ...
        'Indices', idx, 'InputPortWidth', 'nl_N');
end
end

function s = dialogNames(blk)
try
    s = strjoin(fieldnames(get_param(blk, 'DialogParameters')).', ', ');
catch
    s = '(unavailable)';
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

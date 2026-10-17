function w = nl_hann_periodic(N)
%NL_HANN_PERIODIC  Periodic Hann window, w[n] = 0.5 - 0.5*cos(2*pi*n/N), n = 0..N-1.
%
%   Same as hann(N, 'periodic') (Signal Processing Toolbox) and
%   scipy.signal.get_window('hann', N, fftbins=True). Written out explicitly so
%   the model does not depend on the periodic/symmetric option of a library
%   block. At 50 % overlap it sums to exactly 1, so plain overlap-add
%   reconstructs the signal (docs/ALGORITHM.md, Eq. 2 and Eq. 13).
%
%   *** UNTESTED: written without access to MATLAB. Run and fix in MATLAB. ***

n = (0:N-1).';
w = 0.5 - 0.5 * cos(2 * pi * n / N);
end

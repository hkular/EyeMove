function [image_final1, image_final2] = FilteredGratingsV3(PatchSize, SF, ppd, fNyquist, Noise_fLow, Noise_fHigh, gray, whitenoiseContrast, orient, kappa)
% Make a set of filtered images (load white noise filtered in the frequency domain).
% Each image has broadband spatial frequency content, and orientation
% content varying between 0-180 deg.

% taken from MMH April 2020
% adapted HK June 2022
% 2/8/23 HK updated for efficiency


%% make white noise image to filter
% start with a meshgrid
X=-0.5*PatchSize+.5:1:.5*PatchSize-.5; Y=-0.5*PatchSize+.5:1:.5*PatchSize-.5;
[x,y] = meshgrid(X,Y);

whitenoise_sine = (sin(SF/ppd*2*pi*(y.*sin(0*pi/180)+x.*cos(0*pi/180))));
sine_whitenoisecontrast = std(whitenoise_sine(:));

% create white noise
%Make uniform noise, put it into fourrier space, make sf filer
noise = rand(PatchSize,PatchSize)*2-1;
fn_noise = fftshift(fft2(noise));
sfFilter = Bandpass2([PatchSize PatchSize], Noise_fLow/fNyquist, Noise_fHigh/fNyquist);
%Get rid of gibbs ringing artifacts
smoothfilter = fspecial('gaussian', 10, 4);   % make small gaussian blob
sfFilter = filter2(smoothfilter, sfFilter); % convolve smoothing blob w/ s.f. filter
%Bring noise back into real space
filterednoise = real(ifft2(ifftshift(sfFilter.*fn_noise)));
%Scale the contrast of the noise back up (it's lost some in the fourier
%domain) by relating it to the contrast of the grating distractor (before gaussian was applied)
filterednoise = filterednoise*(sine_whitenoisecontrast/(std(filterednoise(:))));
%store it
image = max(0,min(255,gray+gray*(whitenoiseContrast * filterednoise)));

%% Define spectral domain grid/sampling rate etc
% first, define sampling rate and frequency axis
samp_rate_pix = 1;   % samples per pixel, always one.
% get frequency axis
nframes = PatchSize;
nyq = .5*samp_rate_pix;
% step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
freq_step_cpp = samp_rate_pix/nframes;
% x-axis in freq domain, after fftshift
fax = -nyq:freq_step_cpp:nyq-freq_step_cpp;
center_pix = find(abs(fax)==min(abs(fax)));

% next, we're listing the frequencies corresponding to each point in
% the FFT representation, in grid form
[ x_freq, y_freq ] = meshgrid(fax,fax);
x_freq = x_freq(:); y_freq = y_freq(:);
% converting these into a polar format, where angle is orientation and
% magnitude is spatial frequency
[ang_grid,mag_grid] = cart2pol(x_freq,y_freq);
% adjust the angles so that they go from 0-pi in rads
%ang_grid = mod(ang_grid,pi);
ang_grid = mod(ang_grid,pi);
ang_grid = reshape(ang_grid,PatchSize,PatchSize);
mag_grid = reshape(mag_grid,PatchSize,PatchSize);

freq_sd_cpp = 0.005; % sigma for the gaussian fall-off
freq_min_cpp = 0.02;
freq_max_cpp = 0.25;
assert(freq_min_cpp>=freq_sd_cpp*4);

%% Define SF filter in spectral domain
tar_mag = normcdf(mag_grid, freq_min_cpp, freq_sd_cpp).*1-normcdf(mag_grid, freq_max_cpp,freq_sd_cpp);

%% Make a gaussian orientation filter in spectral domain

% adjust for weird 0 deg being vertical thing
if orient >= 90
    input_orient = abs(orient-270);
elseif orient < 90
    input_orient = abs(orient-90);
end

%input_orient = 180-orient;

tar_ang = reshape(circ_vmpdf(ang_grid*2, (input_orient)*pi/180*2, kappa*pi/180*2),PatchSize,PatchSize);
tar_ang = tar_ang./max(tar_ang(:));
% important - make sure that the filter has a constant value at the very center
% of the frequency space. because this value has no angular
% meaning.
tar_ang(center_pix,center_pix) = 0;


%% FFT and filter the image

% apply the filters
image_fft_filt = image.*tar_mag.*tar_ang;

mag = abs(image_fft_filt);

% replace phase values with random numbers between -pi +pi
fake_phase = (rand(size(image_fft_filt))-0.5)*2*pi;

% create the full complex array again
%image_fft_filt_shuff = complex(mag.*sin(fake_phase), mag.*cos(fake_phase)); % flipped sin and cos
image_fft_filt_shuff = complex(mag.*cos(fake_phase), mag.*sin(fake_phase)); 

% back to spatial domain
image_filtered = real(ifft2(fftshift(image_fft_filt_shuff)));

% lost some contrast in the fourier domain, scale it back up
scaling_factor = std(ang_grid(:))/std(image_filtered(:)); % image contrast / current contrast
image_final1 = image_filtered*scaling_factor;
image_final2 = -image_final1;

end



%imshow imagesc
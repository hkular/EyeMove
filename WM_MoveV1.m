%% %%   HK distractor and uncertainty 11/2023
% addpath(genpath('/Applications/Psychtoolbox'))
% Inputs
% p parameter
% info subject info
% nruns: number of runs to execute sequentially
% startRun: run number to start with if interrupted (default is 1)

% Stimulus categories
% Target: full screen gabor orientation w/noise - set size + uncertainty 2 levels
% structured noise
% distractor: present and absent full screen noise mask, saccade point in
% both conditions
% Testing sensory recruitment critic hypothesis that new saccades would
% wipe sensory areas

% Experimental design
% Run duration: XX mins
% Total duration: XX mins or XX mins per session
% Task: orientation full report and distractor change discrimination
%%
function WM_MoveV1(p, info, doET, nruns, startRun)

%% Prepare and collect basic info

%Set Paths
expdir = pwd;
datadir = 'Data';
addpath(pwd);

% set the random seed
rng('default')
rng('shuffle')
t.MySeed = rng; % Save the random seed settings!!

% get time info
info.TheDate = datestr(now,'yymmdd'); %Collect todays date (in t.)
info.TimeStamp = datestr(now,'HHMM'); %Timestamp for saving out a uniquely named datafile (so you will never accidentally overwrite stuff)

%-------------------------------------------------------------------------
KbName('UnifyKeyNames');
if p.MRI == 1 % Scanner projector
    p.start = KbName('t');
    Screen('Preference', 'SkipSyncTests',1);
    p.ccwFast = KbName('b');
    p.ccwSlow = KbName('y');
    p.cwSlow = KbName('g');
    p.cwFast = KbName('r');
    p.keys = [p.ccwFast, p.ccwSlow, p.cwSlow, p.cwFast];
elseif p.environment == 1 && ~p.MRI % Behavior rooms
    p.start = KbName('space');
    Screen('Preference', 'SkipSyncTests',0);
    p.ccwFast = KbName('LeftArrow');
    p.ccwSlow = KbName('UpArrow');
    p.cwSlow = KbName('DownArrow');
    p.cwFast = KbName('RightArrow');
    p.keys = [p.ccwFast, p.ccwSlow, p.cwSlow, p.cwFast];
elseif p.environment == 2  % get realistic size when debugging on Macbook
    p.start = KbName('space');
    Screen('Preference', 'SkipSyncTests',1);
    Screen('preference','Conservevram', 8192);
    p.ccwFast = KbName('LeftArrow');
    p.ccwSlow = KbName('UpArrow');
    p.cwSlow = KbName('DownArrow');
    p.cwFast = KbName('RightArrow');
    p.keys = [p.ccwFast, p.ccwSlow, p.cwSlow, p.cwFast];
end


%% Screen parameters
ScreenNr = 0;
p.ScreenSizePixels = Screen('Rect', ScreenNr);
tmprect = get(0, 'ScreenSize');
computer_res = tmprect(3:4);
if computer_res(1) ~= p.ScreenSizePixels(3) || computer_res(2) ~= p.ScreenSizePixels(4)
    Screen('CloseAll');clear screen;ShowCursor;
    disp('*** ATTENTION *** screensizes do not match''')
end
if p.windowed == 0
    CenterX = p.ScreenSizePixels(3)/2;
    CenterY = p.ScreenSizePixels(4)/2;
else
    CenterX = 1024/2;
    CenterY = 768/2;
end % if windowed
[width, height] = Screen('DisplaySize', ScreenNr); % this is in mm
p.ScreenHeight = height/10; 
p.ViewDistance = 57; % (57 cm is the ideal distance where 1 cm equals 1 visual degree)
p.VisAngle = (2*atan2(p.ScreenHeight/2, p.ViewDistance))*(180/pi); % visual angle of the whole screen
p.ppd = p.ScreenSizePixels(4)/p.VisAngle; % pixels per degree visual angle
p.black=BlackIndex(ScreenNr); p.white=WhiteIndex(ScreenNr);
if p.gammacorrect; p.gray=round((p.black+p.black+p.black+p.white)/4); else; p.gray=round((p.black+p.white)/2);end % darker if gammacorrect
if round( p.gray)==p.white
    p.gray = p.black;
end
p.fNyquist = 0.5*p.ppd;
clear tmprect computer_res 

%% Initialize data files and open
cd(datadir);
if exist(['WM_MoveV1_S', num2str(info.SubNum), '_Main.mat'])
    load(['WM_MoveV1_S', num2str(info.SubNum), '_Main.mat']);
    p.runNum = length(TheData) + 1; % set the number of the current run
    p.startRun = startRun;
    p.nruns = nruns;
    p.TrialNumGlobal = TheData(end).p.TrialNumGlobal;
    p.NumTrials = TheData(end).p.NumTrials;
    p.NumOrientBins = TheData(end).p.NumOrientBins;
    p.OrientBins = TheData(end).p.OrientBins;
    p.Kappa = TheData(end).p.Kappa;
    p.change = TheData(end).p.change;
    p.StartTrial = TheData(end).p.TrialNumGlobal+1;
    p.Block = TheData(end).p.Block+1;
    p.designMat = TheData(end).p.designMat;
    p.trial_cnt_shuffled = TheData(end).p.trial_cnt_shuffled;
else
    p.runNum = 1; %If no data file exists this must be the first run
    p.Block = p.runNum;
    p.TrialNumGlobal = 0;
    p.startRun = startRun;
    p.nruns = nruns;

    %Experimental params required for counterbalancing
    p.NumOrientBins = 4; %must be multiple of the size of your orientation space (here: 180)
    p.OrientBins = reshape(1:180,180/p.NumOrientBins,p.NumOrientBins);
    p.Kappa = [100 5000]; %
    p.Distractor = [1 2]; % absent present
    %----------------------------------------------------------------------
    %COUNTERBALANCING ACT--------------------------------------------------
    %---------------------------------------------------------------------
    TrialStuff = [];
    designMat = fullfact([ 4 2 2]); %  4 ori bins target, 2 distractor present absent, 2 kappa bandwidth
    % replicate for balanced trial nums
    designMat = repmat(designMat,50,1); % 1600 trials total
    % shuffle trials
    trial_cnt = 1:length(designMat);
    trial_cnt_shuffled = Shuffle(trial_cnt);
    for i = 1:length(designMat)        
        trial.orient = randsample(p.OrientBins(:,(designMat(trial_cnt_shuffled(i),2))),1);
        trial.distractor = p.Distractor(designMat(trial_cnt_shuffled(i),2));
        trial.kappa = p.Kappa(designMat(trial_cnt_shuffled(i),3));
        TrialStuff = [TrialStuff trial];
    end
    p.designMat = designMat;
    p.trial_cnt_shuffled = trial_cnt_shuffled;
    if p.shortTrial == 0
        p.NumTrials = 32;% %NOT TRIVIAL! --> must be divisible by MinTrialNum AND by the number of possible iti's (which is 3)
    else
        p.NumTrials = 6;
    end
end
clear designMat trial_cnt trial_cnt_shuffled tmp
cd(expdir); %Back to experiment dir

%% Main Parameters
%Timing Target
t.PhaseReverseFreq = 8; %in Hz, how often gratings reverse their phase
t.PhaseReverseTime = 1/t.PhaseReverseFreq;
t.TargetTime = 4*t.PhaseReverseTime; % 500 ms multiple of Phase reverse time
%Timing Distractor
p.nDistsTrial = 15; % number of different ones to make
t.DistractorTime = 3;% actual distractor time
t.DistFreq = 25; % how fast distractors flip...make it faster 25Hz
t.DistFlipTime = 1/t.DistFreq;
p.nDistFrames = round(t.DistractorTime * t.DistFreq); % how many frames are we flipping through
t.DistArray = [];% pre-randomize frames
for i = 1:(p.nDistFrames/p.nDistsTrial)
    t.DistArray = [t.DistArray;randperm(p.nDistsTrial)']; % randomize frames no repeats
end
%Timing Dot Saccade
t.MinDotTime = 1; %  make it .05 after debugging !!! in s so min 50 ms, stays until confirm fixation
%Timing Other
t.isi1 = 1; %time between memory stimulus and distractor - 0
t.isi2 = 1; %time between distractor and recall probe - 0
t.ResponseTime = 3;
t.BeginFixation = 2; %16 TRs need to be extra (16trs * .8ms)
t.EndFixation = 2;
t.lasttrialiti = .5;
%t.miniti = .7;
t.iti = NaN(p.NumTrials,1);
t.ActiveTrialDur = t.TargetTime+t.isi1+t.MinDotTime+t.DistractorTime+t.isi2+t.ResponseTime; %non-iti portion of trial
t.MeantToBeTime = t.BeginFixation + t.ActiveTrialDur*p.NumTrials + t.EndFixation; % excluding iti

%Stimulus params (general)
p.Smooth_size = round(.75*p.ppd); %size of fspecial smoothing kernel
p.Smooth_sd = round(.4*p.ppd); %smoothing kernel sd
p.PatchSize = round(2*10*p.ppd); %Size of the patch that is drawn on screen location, so twice the radius, in pixels
p.OuterDonutRadius = (10*p.ppd)-(p.Smooth_size/2); %Size of donut outsides, automatically defined in pixels.
p.InnerDonutRadius = (2*p.ppd)+(p.Smooth_size/2); %Size of donut insides, automatically defined in pixels.
p.OuterFixRadius = .2*p.ppd; %outter dot radius (in pixels)
p.InnerFixRadius = p.OuterFixRadius/2; %set to zero if you a donut-hater
p.FixColor = p.black;
p.ResponseLineWidth = 2; %in pixel
p.ResponseLineColor =p.white;
MyPatch = [(CenterX-p.PatchSize/2) (CenterY-p.PatchSize/2) (CenterX+p.PatchSize/2) (CenterY+p.PatchSize/2)];
p.DotSize = .5*p.ppd;
p.Dot = [0 0 p.DotSize p.DotSize];
p.DotColor = [245.0980 36.8980 17.6039]; % red-ish
[X, Y] = meshgrid(MyPatch(1):p.DotSize:MyPatch(3), MyPatch(2):p.DotSize:MyPatch(4));
p.dotPositions = [X(:), Y(:)];
p.DotTol = 2*p.ppd;

%Stimulus params (specific)
p.SF = 2; %spatial frequency in cpd
p.ContrastTarget = .5; %
p.whitenoiseContrast = .5; %
p.distcontrast = [.5 .5]; % to index with TrialStuff.distractor
p.Noise_f_bandwidth = 2;% frequency of the noise bandwidth
p.Noise_fLow = p.SF/p.Noise_f_bandwidth; %Noise low spatial frequency cutoff
p.Noise_fHigh = p.SF*p.Noise_f_bandwidth; %Noise high spatial frequency cutoff

%% window setup and gamma correction
% clock
PsychJavaTrouble;
if p.windowed == 0
    [window, ScreenSize] = Screen('OpenWindow', ScreenNr, p.gray);
else
    % if we're dubugging open a smaller window
    [window, ScreenSize]=Screen('OpenWindow', ScreenNr, p.gray, [0 0 1024 768]);
end
t.ifi = Screen('GetFlipInterval',window);
if p.gammacorrect %
    OriginalCLUT = Screen('LoadClut', window);
    MyCLUT = zeros(256,3); MinLum = 0; MaxLum = 1;
    if strcmp(p.room,'A') % EEG Room
        CalibrationFile = 'LabEEG-05-Jul-2017';
    elseif strcmp(p.room,'B') % Behavior Room B
        CalibrationFile = 'LabB_20-Jul-2022.mat';
    elseif strcmp(p.room,'C') % Behavior Room C !!! check room C
        CalibrationFile = 'LabC-13-Jun-2016.mat';
    elseif strcmp(p.room,'D') % Beahvior room D
        CalibrationFile = 'LabD_20-Jul-2022.mat';
    else
        error('No calibration file specified')
    end
    [gamInverse,dacsize] = LoadCalibrationFileRR(CalibrationFile, expdir, p.GeneralUseScripts);
    LumSteps = linspace(MinLum, MaxLum, 256)';
    MyCLUT(:,:) = repmat(LumSteps, [1 3]);
    MyCLUT = map2map(MyCLUT, repmat(gamInverse(:,3),[1 3])); %Now the screen output luminance per pixel is linear!
    Screen('LoadCLUT', window, MyCLUT);
    clear CalibrationFile gamInverse
end

if p.debug == 0
    HideCursor;
end

for b = startRun:nruns % block loop
    %% preallocate in block loop
    data = struct();
    % behavior
    data.Accuracy = NaN(p.NumTrials, 1);
    data.Response = NaN(p.NumTrials, 1);
    data.RTresp = NaN(p.NumTrials, 1);
    data.TestOrient = randsample(1:180,p.NumTrials,true);
    data.DotSample = randsample(1:size(p.dotPositions,1), p.NumTrials, true);
    data.DistResp = NaN(p.NumTrials,1);
    data.DistReact = NaN(p.NumTrials,1);
    % preallocate cells so get multiple values per trial
    data.Trajectory = cell(p.NumTrials, 1);

    %% Make target stimuli
    % start with a meshgrid
    X=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5; Y=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5;
    [x,y] = meshgrid(X,Y);
    % make a donut with gaussian blurred edge
    donut_out = x.^2 + y.^2 <= (p.OuterDonutRadius)^2;
    donut_in = x.^2 + y.^2 >= (p.InnerDonutRadius)^2;
    donut = donut_out.*donut_in;
    donut = filter2(fspecial('gaussian', p.Smooth_size, p.Smooth_sd), donut);

    % 4D array - target position, x_size, y_size, numtrials
    % initialize with middle grey (background color), then fill in a
    % 1 or 2 as needed for each trial.
    TargetsAreHere = ones(p.PatchSize,p.PatchSize,2) * p.gray; % last dimension 2 phases
    startTrialThisRun = (p.NumTrials * p.runNum) - p.NumTrials + 1;
    % call function that creates filtered gratings
    [image_final1, image_final2] = FilteredGratingsV3(p.PatchSize, p.SF, p.ppd, p.fNyquist, p.Noise_fLow, p.Noise_fHigh, p.gray, p.whitenoiseContrast, TrialStuff(startTrialThisRun).orient, TrialStuff(startTrialThisRun).kappa);
    %Make it a donut
    stim_phase1 = image_final1.*donut;
    stim_phase2 = image_final2.*donut;
    %Give the grating the right contrast level and scale it
    TargetsAreHere(:,:,1) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase1)));
    TargetsAreHere(:,:,2) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase2)));

    %% make saccade dot stimuli
    dot = p.dotPositions(data.DotSample(startTrialThisRun),1:2);
    dotLocation = [dot(1)-p.OuterFixRadius dot(2)-p.OuterFixRadius dot(1)+p.OuterFixRadius dot(2)+p.OuterFixRadius];   

    %% make distractor stimuli - same size as target but pure noise

    % now make a matrix with with all my distractors for all my trials
    DistractorsAreHere = NaN(p.PatchSize,p.PatchSize, p.nDistsTrial); % last dimension makes it dynamic
    distractor_sine = (sin(p.SF/p.ppd*2*pi*(y.*sin(0*pi/180)+x.*cos(0*pi/180))));
    sine_contrast = std(distractor_sine(:));
    for num = 1 : p.nDistsTrial
        %Make uniform noise, put it into fourrier space, make sf filer
        noise = rand(p.PatchSize,p.PatchSize)*2-1;
        fn_noise = fftshift(fft2(noise));
        sfFilter = Bandpass2([p.PatchSize p.PatchSize], p.Noise_fLow/p.fNyquist, p.Noise_fHigh/p.fNyquist);
        %Get rid of gibbs ringing artifacts
        smoothfilter = fspecial('gaussian', 10, 4);   % make small gaussian blob
        sfFilter = filter2(smoothfilter, sfFilter); % convolve smoothing blob w/ s.f. filter
        %Bring noise back into real space
        filterednoise = real(ifft2(ifftshift(sfFilter.*fn_noise)));
        %Scale the contrast of the noise back up (it's lost some in the fourier
        %domain) by relating it to the contrast of the grating distractor (before gaussian was applied)
        current_noise_contrast = std(filterednoise(:));
        scaling_factor = sine_contrast/current_noise_contrast;
        filterednoise = filterednoise*scaling_factor;
        %Make it a donut
        filterednoise_phase = filterednoise .* donut;
        DistractorsAreHere(:,:,num) = max(0,min(255,p.gray+p.gray*(p.distcontrast(TrialStuff(startTrialThisRun).distractor) * filterednoise_phase)));
    end %for ndists


% %% Setup Eye-tracking %
% if doET
%     %%% ET %%%
%     connected = Eyelink('Initialize','PsychEyelinkDispatchCallback');
%     
%     %check if it worked
%     if Eyelink('IsConnected') == 1 && connected == 0 %two ways of checking if it is connected
%         et.etOn = 1;
%     else
%         et.etOn = 0;
%         msg = 'No eye tracker detected. Should I try to continue? (y/n)';
%         DrawFormattedText(win, msg, 'center', 'center', 255);
%         Screen('Flip',win);
%         [~, keycode] = KbWait(-1);
%         if any(strcmp(KbName(keycode),'n'))
%             error('Execution terminated by user')
%         elseif any(strcmp(KbName(keycode),'y'))
%             if (Eyelink('Initialize') ~= 0)
%                 et.etOn = 0;
%                 error('Sorry, cannot connect')
%             elseif (Eyelink('Initialize') == 0)
%                 et.etOn = 1;
%             end
%         end
%     end
%     
%     % which data to save
%     status = Eyelink('Command',...
%         'link_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,HREF,PUPIL,STATUS,INPUT');
%     Eyelink('Command', 'sample_rate = 1000');
%     
%     % tell eyelink current screen resolution
%     Eyelink('command','screen_pixel_coords = %ld %ld %ld %ld',0, 0,w-1,h-1);
%     
%     % setting up with the Eyelink routine
%     el = EyelinkInitDefaults(win);
%     EyelinkDoTrackerSetup(el);
% 
%     %--------------------------------------------------------------------------
%     % Allocate space to save eyetracker violations
%     %--------------------------------------------------------------------------
%     stim = struct(); % this will be used at the end to draw plots
% 
%     stim.gazeViolationIdx = NaN([10,1]);
%     stim.gazeViolationType = NaN([10,1]);
%     stim.gazeViolationTypeDisplay = NaN([10,1]);
%     stim.gazeViolationEvents = NaN([10,1]);
%     stim.gazeViolationDisplay = NaN([10,1]);
%     stim.violationOnset = NaN([10,1]);
% 
%     % some extra parameters to detect fixation violation
%     p.fixTol = 2*p.ppd;
%     p.violationMaxDur = .05;
%     p.eyeDirectFB = 1; % give live feedback of gaze position
% end


    %% Welcome and wait for trigger
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
     
    Screen(window,'TextSize',30); 
    Screen('DrawText',window, 'Fixate. Press spacebar to begin.', CenterX-300, CenterY-75,p.black); 
    Screen('Flip', window); 
    FlushEvents('keyDown'); %First discard all characters from the Event Manager queue.
    ListenChar(2);
    % just sittin' here, waitin' on my trigger...
    while 1
        [keyIsDown, secs, keyCode] = KbCheck([-1]); % KbCheck([-1])
        if keyCode(KbName('space'))
            t.StartTime = GetSecs;
            break; %let's go!
        end
    end
    FlushEvents('keyDown');

    GlobalTimer = 0; %this timer keeps track of all the timing in the experiment. TOTAL timing.
    TimeUpdate = t.StartTime; %what time is it now?
    % present begin fixation
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen('Flip', window);
    %TIMING!:
    GlobalTimer = GlobalTimer + t.BeginFixation;
    TimePassed = 0; %Flush the time the previous event took
    while (TimePassed<t.BeginFixation) %For as long as the cues are on the screen...
        TimePassed = (GetSecs-TimeUpdate);%And determine exactly how much time has passed since the start of the expt.
    end
    TimeUpdate = TimeUpdate + t.BeginFixation;

% %% SET UP EYETRACKING -- FILENAME & START RECORDING
% if doET
%     % filename
%     edf_fn = sprintf('OT%s%02d.edf',p.subName,p.currBlock);
%     toRename_edf = fullfile(data_dir, sprintf('%s_S%s_B%02d.edf',expName,p.subName,p.currBlock));
%     status = Eyelink('OpenFile', edf_fn);
%     
%     % timestamps of exp info
%     % Eyelink('Message',sprintf('BLOCK_%02d_STARTED',b));
%     Eyelink('Message', ['FILENAME: ', toRename_edf]);
%     Eyelink('Message', ['SUBJECT: ', p.subName]);
%     Eyelink('Message', ['BLOCK: ', sprintf('%02d', p.currBlock)]);
%     Eyelink('Message', ['SYSTEM DATE AND TIME: ', ...
%         datestr(now, 'dd-mm-yyyy HH:MM:SS')]);
% 
%     % Check which eye is available for gaze-contingent drawing. Returns 0 (left), 1 (right) or 2 (binocular)
%     stim.eyeUsed = Eyelink('EyeAvailable');
%     
%     % START RECORDING
%     Eyelink('StartRecording');
%     Eyelink('Message','SESSION_STARTED');
% end
%% Start trial loop    
    for n = 1:p.NumTrials
        t.TrialStartTime(n) = GlobalTimer; %Get the starttime of each single block (relative to experiment start)
        TimeUpdate = t.StartTime + t.TrialStartTime(n);
        p.TrialNumGlobal = p.TrialNumGlobal+1;
        %if doET; Eyelink('Message',['TRIAL_', num2str(p.TrialNumGlobal), '.START']); end %%% ET %%%


        %% Target rendering
        tic
        for revs = 1:t.TargetTime/t.PhaseReverseTime
            StimToDraw = Screen('MakeTexture', window, TargetsAreHere(:,:,rem(revs,2)+1));
            Screen('DrawTexture', window, StimToDraw, [], MyPatch, [], 0);
            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('DrawingFinished', window);
            Screen('Flip', window);
            %TIMING!:
            GlobalTimer = GlobalTimer + t.PhaseReverseTime;
            ReversalTimePassed = 0; %Flush time passed.
            % Wait the time!
            while (ReversalTimePassed<t.PhaseReverseTime) %As long as the stimulus is on the screen...
                ReversalTimePassed = (GetSecs-TimeUpdate);
            end
            TimeUpdate = TimeUpdate + t.PhaseReverseTime;
        end
        toc

        %% delay 1
        tic
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
        %TIMING!:
        GlobalTimer = GlobalTimer + t.isi1;
        delay1TimePassed = (GetSecs-TimeUpdate);
        while (delay1TimePassed<t.isi1) %As long as the stimulus is on the screen...
            delay1TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
        end
        TimeUpdate = TimeUpdate + t.isi1; %Update Matlab on what time it is.
        toc
        %% Draw Saccade Dot
        tic
        ETConfirm = 0;
        DotElapse = 0; % flush time elapsed
        while ETConfirm == 0 && DotElapse < t.MinDotTime
            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius]);
            Screen('FillOval', window, p.DotColor, dotLocation);
            Screen('Flip', window);
            if doET > 0
                % Get gaze position
                [x, y] = Eyelink('GetGazePos');

                % Check if gaze position coincides with the dot center
                withinTol = sqrt((x - ((dotLocation(1) + dotLocation(3)) / 2))^2 + (y - ((dotLocation(2) + dotLocation(4)) / 2))^2) <= p.fixTol;        
                if withinTol
                    ETConfirm = 1;
                end%if x
            end %if ET
            DotElapse = (GetSecs - TimeUpdate);
        end %while ET
        TimeUpdate = TimeUpdate + DotElapse;
        toc
        %% Distractor
        tic
        for d = 1:p.nDistsTrial
            DistToDraw(d) = Screen('MakeTexture', window, DistractorsAreHere(:,:,d));
        end %for

        react = NaN;
        dist_start = GetSecs;
        for k = 1:round(t.DistractorTime * t.DistFreq)
            Screen('DrawTexture', window, DistToDraw(t.DistArray(k)), [], MyPatch, [],0);
            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('DrawingFinished', window);
            Screen('Flip', window);
            GlobalTimer = GlobalTimer + t.DistFlipTime;
            [keyIsDown, secs, keyCode] = KbCheck(-1);
            if keyIsDown == 1
                % buttons
                if keyCode(p.ccwFast) %BIG step CCW
                    data.DistResp(n) = p.ccwFast;
                    react = secs - dist_start;
                elseif keyCode(p.ccwSlow) %small step CCW
                    data.DistResp(n) = p.ccwSlow;
                    react = secs - dist_start;
                elseif keyCode(p.cwSlow) %small step CW
                    data.DistResp(n) = p.cwSlow;
                    react = secs - dist_start;
                elseif keyCode(p.cwFast) %BIG step CW
                    data.DistResp(n) = p.cwFast;
                    react = secs - dist_start;
                end
            end
            FlipTimePassed = 0; %Flush time passed.
            % Wait the time!
            while (FlipTimePassed<t.DistFlipTime) %As long as the stimulus is on the screen...
                FlipTimePassed = (GetSecs-TimeUpdate);
            end%while
            TimeUpdate = TimeUpdate + t.DistFlipTime;
        end%for
        data.DistReact(n) = react;
        Screen('Close', [DistToDraw]);
        clear d DistToDraw 
        toc
        %% isi2
        tic
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
        %TIMING!:
        GlobalTimer = GlobalTimer + t.isi2;
        delay2TimePassed = (GetSecs-TimeUpdate);
        while (delay2TimePassed<t.isi2) %As long as the stimulus is on the screen...
            delay2TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
            % keep listening for key press response to distractor
            [keyIsDown, secs, keyCode] = KbCheck(-1);
            if keyIsDown == 1
                % buttons
                if keyCode(p.ccwFast) %BIG step CCW
                    data.DistResp(n) = p.ccwFast;
                    react = secs - dist_start;
                elseif keyCode(p.ccwSlow) %small step CCW
                    data.DistResp(n) = p.ccwSlow;
                    react = secs - dist_start;
                elseif keyCode(p.cwSlow) %small step CW
                    data.DistResp(n) = p.cwSlow;
                    react = secs - dist_start;
                elseif keyCode(p.cwFast) %BIG step CW
                    data.DistResp(n) = p.cwFast;
                    react = secs - dist_start;
                end
            end
        end
            TimeUpdate = TimeUpdate + t.isi2; %Update Matlab on what time it is.
            data.DistReact(n) = react;
        toc
        %% response window
 
        % full report spin a line, in quadrant we are probing
        resp_start = GetSecs;
        test_orient = data.TestOrient(n);
        orient_trajectory = [test_orient];
        InitX = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * cos(test_orient*pi/180)+CenterX));
        InitY = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * sin(test_orient*pi/180)-CenterY));
        Screen('BlendFunction', window, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
        Screen('DrawLines', window, [2*CenterX-InitX, InitX; 2*CenterY-InitY, InitY], p.ResponseLineWidth, p.ResponseLineColor,[],1);
        Screen('BlendFunction', window, GL_ONE, GL_ZERO);
        Screen('FillOval', window, p.gray, [CenterX-(p.InnerDonutRadius-p.Smooth_size/2) CenterY-(p.InnerDonutRadius-p.Smooth_size/2) CenterX+(p.InnerDonutRadius-p.Smooth_size/2) CenterY+(p.InnerDonutRadius-p.Smooth_size/2)]);
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window,[],1);
        GlobalTimer = GlobalTimer + t.ResponseTime;
        react = NaN;
        RespTimePassed = GetSecs-resp_start; %Flush time passed.
        while RespTimePassed<t.ResponseTime  %As long as no correct answer is identified
            RespTimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
            [keyIsDown, secs, keyCode] = KbCheck(-1);
            % buttons
            if keyCode(p.ccwFast) %BIG step CCW
                test_orient = rem(test_orient+2+1440,180);
                react = secs - resp_start;
                % alternate way of getting RT
          p.DTBins = [-60:-30;-15:15; 30:60; 75:105];      % RT(n,tt) = secs-resp_start;
            elseif keyCode(p.ccwSlow) %small step CCW
                test_orient = rem(test_orient+.5+1440,180);
                react = secs - resp_start;
            elseif keyCode(p.cwSlow) %small step CW
                test_orient = rem(test_orient-.5+1440,180);
                react = secs - resp_start;
            elseif keyCode(p.cwFast) %BIG step CW
                test_orient = rem(test_orient-2+1440,180);
                react = secs - resp_start;
            elseif keyCode(KbName('ESCAPE')) % If user presses ESCAPE, exit the program.
                Screen('CloseAll');
                ListenChar(1);
                if exist('OriginalCLUT','var')
                    if exist('ScreenNr','var')
                        Screen('LoadCLUT', ScreenNr, OriginalCLUT);
                    else
                        Screen('LoadCLUT', 0, OriginalCLUT);
                    end
                end
                error('User exited program.');
            end
            test_orient(test_orient==0)=180;
            orient_trajectory = [orient_trajectory test_orient];
            UpdatedX = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * cos(test_orient*pi/180)+CenterX));
            UpdatedY = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * sin(test_orient*pi/180)-CenterY));
            Screen('BlendFunction', window, GL_ONE, GL_ZERO);
            Screen('FillRect', window, p.gray);
            Screen('BlendFunction', window, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
            Screen('DrawLines', window, [2*CenterX-UpdatedX, UpdatedX; 2*CenterY-UpdatedY, UpdatedY], p.ResponseLineWidth, p.ResponseLineColor, [], 1);
            Screen('BlendFunction', window, GL_ONE, GL_ZERO);
            Screen('FillOval', window, p.gray, [CenterX-(p.InnerDonutRadius-p.Smooth_size/2) CenterY-(p.InnerDonutRadius-p.Smooth_size/2) CenterX+(p.InnerDonutRadius-p.Smooth_size/2) CenterY+(p.InnerDonutRadius-p.Smooth_size/2)]);
            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius]);
            Screen('Flip', window, [], 1,[], []);
        end
        FlushEvents('keyDown'); %First discard all characters from the Event Manager queue
        data.Response(n) = test_orient;
        data.RTresp(n) = react;
        data.Trajectory{n} = orient_trajectory; %
        %if no keys pressed NaN
        if data.Response(n) == data.TestOrient(n)
            data.Response(n) = NaN;
        end
        TimeUpdate = TimeUpdate + t.ResponseTime; %Update Matlab on what time it is.
       
        %% iti
       
        Screen('FillRect',window,p.gray);
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        if n < p.NumTrials
            DrawFormattedText(window,'Press spacebar for the next trial.',CenterX-170,CenterY-40,p.white);
        end
        Screen('DrawingFinished', window);
         Screen ('Flip', window);
        % Make things during ITI must be less than <2sec shortest iti

        if  n < p.NumTrials%p.TrialNumGlobal <length(TrialStuff) if we have another trial
            % TARGET for next trial
            %TargetsAreHere = ones(p.PatchSize,p.PatchSize,2) * p.gray;
            [image_final1, image_final2] = FilteredGratingsV3(p.PatchSize, p.SF, p.ppd, p.fNyquist, p.Noise_fLow, p.Noise_fHigh, p.gray, p.whitenoiseContrast, TrialStuff(p.TrialNumGlobal+1).orient, TrialStuff(p.TrialNumGlobal+1).kappa);
            stim_phase1 = image_final1.*donut;
            stim_phase2 = image_final2.*donut;
            TargetsAreHere(:,:,1) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase1)));
            TargetsAreHere(:,:,2) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase2)));
            % DOT for next trial
            dot = p.dotPositions(data.DotSample(p.TrialNumGlobal+1),1:2);
            dotLocation = [dot(1)-p.OuterFixRadius dot(2)-p.OuterFixRadius dot(1)+p.OuterFixRadius dot(2)+p.OuterFixRadius];   
            % DISTRACTOR for next trial
            for num = 1 : p.nDistsTrial
                noise = rand(p.PatchSize,p.PatchSize)*2-1;
                fn_noise = fftshift(fft2(noise));
                %sfFilter = Bandpass2([p.PatchSize p.PatchSize], p.Noise_fLow/p.fNyquist, p.Noise_fHigh/p.fNyquist);
                %smoothfilter = fspecial('gaussian', 10, 4);
                %sfFilter = filter2(smoothfilter, sfFilter);
                filterednoise = real(ifft2(ifftshift(sfFilter.*fn_noise)));
                current_noise_contrast = std(filterednoise(:));
                scaling_factor = sine_contrast/current_noise_contrast;
                filterednoise = filterednoise*scaling_factor;
                filterednoise_phase = filterednoise .* donut;
                DistractorsAreHere(:,:,num) = max(0,min(255,p.gray+p.gray*(p.distcontrast(TrialStuff(p.TrialNumGlobal+1).distractor) * filterednoise_phase)));
            end%for
        end %if we have another trial coming up

        % after finished loading next trial, wait for key press to continue
        if n < p.NumTrials
            while 1
                [keyIsDown, ~, keyCode] = KbCheck(-1); % KbCheck([-1])
                if keyCode(KbName('space'))
                    t.iti(n) = GetSecs - TimeUpdate;
                    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
                    Screen('Flip', window);
                    break; %next block
                end
            end
            FlushEvents('keyDown');
        else
            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('Flip', window);
            TimePass = 0;
            while (TimePass<t.lasttrialiti) %For as long as the cues are on the screen...
                TimePass = (GetSecs-TimeUpdate);%And determine exactly how much time has passed since the start of the expt.
            end
            t.iti(n) = t.lasttrialiti;
        end
        TimeUpdate = TimeUpdate + t.iti(n);  
        GlobalTimer = GlobalTimer + t.iti(n);
      
    end %end of experimental trial loop

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% END OF TRIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %----------------------------------------------------------------------
    %LOOK AT BEHAVIORAL PERFOPRMANCE---------------------------------------
    %----------------------------------------------------------------------
    targets_were = [TrialStuff(p.TrialNumGlobal+1-p.NumTrials:p.TrialNumGlobal).orient]';
    acc(1,:) = abs(targets_were-data.Response);
    acc(2,:) = abs((360-(acc(1,:)*2))/2);
    acc(3,:) = 360-(acc(1,:));
    acc = min(acc);
    acc = acc';

    %Add minus signs back in
    acc(mod(targets_were-acc,360)==data.Response)=-acc(mod(targets_were-acc,360)==data.Response);
    acc(mod((targets_were+180)-acc,360)==data.Response)=-acc(mod((targets_were+180)-acc,360)==data.Response);
    data.Accuracy = acc;
    blockStr = ['Finished block ' num2str(b) ' out of ' num2str(nruns)];
    feedbackStr = [blockStr sprintf('\n') 'Press the spacebar to continue'];
   
    %----------------------------------------------------------------------
    %SAVE OUT THE DATA-----------------------------------------------------
    %----------------------------------------------------------------------
    cd(datadir); %Change the working directory back to the experimental directory
    if exist(['WM_MoveV1_S', num2str(info.SubNum), '_Main.mat'])
        load(['WM_MoveV1_S', num2str(info.SubNum), '_Main.mat']);
    end
    %First I make a list of variables to save:
    TheData(p.runNum).info = info;
    TheData(p.runNum).t = t;
    TheData(p.runNum).p = p;
    TheData(p.runNum).data = data;
    eval(['save(''WM_MoveV1_S', num2str(info.SubNum),'_Main.mat'', ''TheData'', ''TrialStuff'', ''-V7.3'')']);
    cd(expdir)

    FlushEvents('keyDown');

    clear TargetsAreHere DistractorsAreHere

    % final fixation and feedback:
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen('Flip', window);
    % may need to change spacing
    DrawFormattedText(window,feedbackStr,CenterX-200,CenterY,p.white);
    Screen('Flip',window);

    while 1
        [keyIsDown, secs, keyCode] = KbCheck([-1]); % KbCheck([-1])
        if keyCode(KbName('space'))
            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('Flip', window);
            break; %next block
        end
    end
    FlushEvents('keyDown');


    GlobalTimer = GlobalTimer + t.EndFixation;

    closingtime = 0; resp = 0;

    while closingtime < t.EndFixation
        closingtime = GetSecs-TimeUpdate;
        ListenChar(1); %Unsuppressed keyboard mode
        if CharAvail
            [press] = GetChar;
            if strcmp(press,'1')
                resp = str2double(press);
            end
        end
    end
    t.EndTime = GetSecs; %Get endtime of the experiment in seconds
    t.TotalExpTime = (t.EndTime-t.StartTime); %Gets the duration of the total run.
    t.TotalExpTimeMins = t.TotalExpTime/60; %TOTAL exp time in mins including begin and end fixation.
    t.GlobalTimer = GlobalTimer; %Spits out the exp time in secs excluding begin and end fixation.

    p.runNum = p.runNum+1;

    clear acc
end  % end of block loop
%----------------------------------------------------------------------
%WINDOW CLEANUP--------------------------------------------------------
%----------------------------------------------------------------------
Screen('CloseAll');
if exist('OriginalCLUT','var')
    if exist('ScreenNr','var')
        Screen('LoadCLUT', ScreenNr, OriginalCLUT);
    else
        Screen('LoadCLUT', 0, OriginalCLUT);
    end
end
clear screen
ListenChar(1);
ShowCursor;

end


%% saccade drawing graveyard
%     tempRect = [0 0 .5*p.ppd .5*p.ppd];
%     
%     
%     distRect = CenterRectOnPoint(tempRect, ...
%             CenterX + cosd(thisAng)*5,...
%             CenterY - sind(thisAng)*5);
%     
%     thisAng = randi(360); % randomly choose angular location for dot
%     DotLoc = [CenterX - cosd(thisAng)*data.DotEccen(startTrialThisRun),...
%         CenterY - sind(thisAng)*data.DotEccen(startTrialThisRun),...
%         CenterX + cosd(thisAng)*data.DotEccen(startTrialThisRun),...
%         CenterY + sind(thisAng)*data.DotEccen(startTrialThisRun)];
% 
%  circleLocation = [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius];
%     
% 
% % Calculate the visual angle offset in pixels
% visualAngleOffset = 4 * p.ppd;
% 
% % Move the circle to the desired visual angle
% finalX = rotatedX +5;%+ visualAngleOffset;
% finalY = rotatedY;
% 
% % move circle to desired visual angle
% circleLocation = [circleLocation(1)+visualAngleOffset circleLocation(2) circleLocation(3)+visualAngleOffset circleLocation(4)];
% 
% % Calculate rotated coordinate
% rotatedX = CenterX + (circleLocation(1) - CenterX) * cosd(thisAng) - (circleLocation(2) - CenterY) * sind(thisAng);
% %rotatedX = circleLocation(1) - 
% rotatedY = CenterY + (circleLocation(1) - CenterX) * sind(thisAng) + (circleLocation(2) - CenterY) * cosd(thisAng);
% 
% 
% % Update circle location
% newCircleLocation = [rotatedX rotatedY rotatedX + (circleLocation(3) - circleLocation(1)) rotatedY + (circleLocation(4) - circleLocation(2))];
% 
% 
% fixLoc = [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius];
   
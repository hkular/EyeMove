% 1/18/22 SP
% delayed match-to-sample task in foveal and peripheral visual field

% 4/26/22 Edited for Orca from Hourglass
% Orca uses a new set of shape stimulus (VCS) that is circular 1-D
% has shorter delay periods (3.5 s)
% has distractor / no-distractor trials
% basically all conditions we want to test thrown in together so learning
% effect is minimized
% distractors blink on and off during delay period for more interference
% trying just one distractor for now but can flash different distractor
% every time if there's need for even more interference
% response widow is limited to 4s
% Eyelink setup screen turns on after every block, but the experimenter
% shoud validate after every block & calibrate after every 4 blocks (or
% calibrate as often as needed if the subject moves away from the chinrest)

% need EyelinkDevKit https://www.sr-support.com/showthread.php?tid=13

% 5/26/22 Edited from pilot
% shorter target duration
% shorter and more frequent flashes of distractors during delay period
% re-run script for every block
% close window and close all textures after every block
% add trial-by-trial feedback on eye movement?
% for prac: make it more extensive
% longer ITI?

% 5/31/22 Fixation training for Orca
% Give trial-by-trial feedback on eye movement
% Trial aborts if subject moves their eyes away from fixation and goes onto
% the next trial.
% Unlimited number of trials - experiment goes on until manually stopped.
% Make an option to have foveal/peripheral mixed or only f or p trials
% Make it possible to calibrate/validate after every trial
% Save and plot data for at which point subject moved their eyes and how
% many for each condition
% Save out eyetracking data only once for each run

function Orca_ETtrain()
clearvars
Screen('CloseAll');
% dbstop if error

%% Basic info
% seed the generator
p.rng_settings = rng('shuffle'); % Save the random seed settings!!

% Get timing info
t.TheDate = datestr(now,'yymmdd'); %Collect todays date (in t.)
t.TimeStamp = datestr(now,'HHMM'); %Timestamp for saving out a uniquely named datafile (so you will never accidentally overwrite stuff)

% EXPERIMENT INFO
% {'Subject Number','Debug?','Eye-tracking?','Condition','Block'};
expInfo = {'00','0','1','0','1'};
p.subName = char(expInfo{1});         % get subject's initials from user
p.prac = 0; % give feedback for practice
p.debug = str2double(expInfo{2}); % debuggin mode: smaller screen, show mouse and listen to keyboard
doET = str2double(expInfo{3});
p.targCond = str2double(expInfo{4}); % 0: both F and P, 1: only F, 2: only P
p.currBlock = str2double(expInfo{5}); % block number in case there are multiple runs of training

% define data & subject directory to load BH info and save data
expName = 'Orca_ETtrain';
data_dir = fullfile(pwd, ['Data_' expName]);
if ~exist(data_dir,'dir'); mkdir(data_dir); end

% set base name for data file
baseName=[expName '_S' p.subName '_B' sprintf('%02d',p.currBlock) '_' num2str(t.TheDate)];

% check if data file already exists
if exist(fullfile(sub_dir, [baseName '.mat']),'file') 
    msgbox('File already exists!', 'modal')
    return
end

% set path where pre-made images live
p.imagedir = fullfile(pwd,'StimulusImages/VCSshapes');

%% RESPONSE PARAMS
KbName('UnifyKeyNames');

% % key set-up
p.keyEscape = KbName('q');
p.keyEnd = KbName('e');
p.keyStart=KbName('space');
% p.keyConfirm=KbName('space');

% keys for adjusting test stim -- do big steps, small steps
p.keyBigLeft = KbName('m');
p.keySmallLeft = KbName(',<');
p.keySmallRight = KbName('.>');
p.keyBigRight = KbName('/?');

% how much steps is it going to move? (0-360 space)
p.bigStep = 5;
p.smallStep = 1;

%% Screen Setup
% Select screen with maximum id for output window:
screenid = max(Screen('Screens'));
if ~p.debug
    %     Screen('Preference', 'SkipSyncTests', 1);
    [win,rect] = Screen(screenid, 'OpenWindow', 128);
else
    Screen('Preference', 'SkipSyncTests', 1);
    [win,rect] = Screen(screenid, 'OpenWindow', 128, [0 0 1000 800]);
    %     [win,rect] = Screen(screenid, 'OpenWindow', 128);
end
Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
[w, h]=Screen('WindowSize', win);
p.w = w; p.h = h;
cx = w/2; cy = h/2;
p.xCenter = cx; p.yCenter = cy;
p.ScreenSizePixels = rect;

if ~p.debug
    ListenChar(-1);
    HideCursor(screenid);
end

% Serences lab bhv spec -- EEG booth
p.viewingdist = 67; %cm
p.monitorwidth = 39; %cm
p.monitorheight = 29.5; %cm

fieldwidth = atan( (p.monitorwidth / 2) / p.viewingdist ) * 2; %width of visual field in radian
fieldheight = atan( (p.monitorheight / 2) / p.viewingdist ) * 2; %height of visual field in radian

% convert to degrees
fieldwidth_deg = fieldwidth * 180 / pi;
fieldheight_deg = fieldheight * 180 / pi;
p.ppd = round(( (w / fieldwidth_deg) + (h / fieldheight_deg) ) / 2);

% get max white and black
p.white = WhiteIndex(win);
p.black = BlackIndex(win);
p.grey = (p.white+p.black)/2;
p.grey = 128;
Screen(win, 'FillRect', p.grey);

%% Temporal Params %%

t.stimDur = 0.15; % WM target stimulus duration
t.maskDur = 0.05;
t.distDur = 0.15; % on-off for 0.3 sec
t.delay = 3.45; % delay until response can be made
t.nDistFlip = floor(t.delay/t.distDur/2); % how many times distractors are shown within the delay
t.respDur = 4; % response window 4s after delay
if p.prac; t.respDur = 7; end % response window 7s for practice
t.ITI = 2; % post-response duration
t.trialDur = t.stimDur+t.delay+t.respDur+t.ITI; % 9.65s

t.BlankStart = 2; % 2s before starting the first trial

%% Visual Params %%
%%% define cue %%%
p.fixSize_pix = 0.2*p.ppd; % diameter
fixationDot = [0 0 p.fixSize_pix p.fixSize_pix];
p.fixationDot = CenterRect(fixationDot, p.ScreenSizePixels);

% define foveal & peripheral target size -- this should be easy to adjust based on accuracy
% setting placeholders for now
p.stimImgSize = [3 5].*p.ppd; % pix
p.stimEccentricity = [0 7].*p.ppd; % pix; from center of image to center of screen

% keeping the cortical magnification equation for reference
% M_db = (0.065*[0 7] + 0.054); % this gives us cortical mm at this deg

%% Setup Eye-tracking %
if doET
    %%% ET %%%
    connected = Eyelink('Initialize','PsychEyelinkDispatchCallback');
    
    %check if it worked
    if Eyelink('IsConnected') == 1 && connected == 0 %two ways of checking if it is connected
        et.etOn = 1;
    else
        et.etOn = 0;
        msg = 'No eye tracker detected. Should I try to continue? (y/n)';
        DrawFormattedText(win, msg, 'center', 'center', 255);
        Screen('Flip',win);
        [~, keycode] = KbWait(-1);
        if any(strcmp(KbName(keycode),'n'))
            error('Execution terminated by user')
        elseif any(strcmp(KbName(keycode),'y'))
            if (Eyelink('Initialize') ~= 0)
                et.etOn = 0;
                error('Sorry, cannot connect')
            elseif (Eyelink('Initialize') == 0)
                et.etOn = 1;
            end
        end
    end
    
    % which data to save
    status = Eyelink('Command',...
        'link_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,HREF,PUPIL,STATUS,INPUT');
    Eyelink('Command', 'sample_rate = 1000');
    
    % tell eyelink current screen resolution
    Eyelink('command','screen_pixel_coords = %ld %ld %ld %ld',0, 0,w-1,h-1);
    
    % setting up with the Eyelink routine
    el = EyelinkInitDefaults(win);
    EyelinkDoTrackerSetup(el);

    %--------------------------------------------------------------------------
    % Allocate space to save eyetracker violations
    %--------------------------------------------------------------------------
    stim = struct(); % this will be used at the end to draw plots

    stim.gazeViolationIdx = NaN([10,1]);
    stim.gazeViolationType = NaN([10,1]);
    stim.gazeViolationTypeDisplay = NaN([10,1]);
    stim.gazeViolationEvents = NaN([10,1]);
    stim.gazeViolationDisplay = NaN([10,1]);
    stim.violationOnset = NaN([10,1]);

    % some extra parameters to detect fixation violation
    p.fixTol = 2*p.ppd;
    p.violationMaxDur = .05;
    p.eyeDirectFB = 1; % give live feedback of gaze position
end

%% Instruction
% show instruction
text_inst = sprintf('Remember the first shape and match the second shape to the first one,\nby using the four keys to adjust the image.\n\nPress SPACE to continue.');
if p.prac % letting you know this is a practice run
    text_inst = [text_inst '\n\nThis is a practice and feedback will be given after each trial.'];
else
    text_inst = [text_inst '\n\nThis is the actual task and feedback will NOT be given.'];
end
text_size = .8; % degree
text_size_pix = floor(text_size*p.ppd);
Screen('TextSize', win, text_size_pix);
DrawFormattedText(win, text_inst, 'center', cy - text_size_pix*1.5, p.white);
Screen(win, 'Flip');

% wait for trigger
while 1
    [keyIsDown, ~, keyCode] = KbCheck();
    if keyIsDown == 1 && keyCode(p.keyStart)==1
        break
    elseif keyIsDown == 1 && keyCode(p.keyEscape)==1
        ShowCursor;
        ListenChar(0);
        Screen('CloseAll');
        return
    end
end
Screen('FillOval', win, p.black, p.fixationDot);
Screen(win, 'Flip');
WaitSecs(2);

%% SET UP EYETRACKING -- FILENAME & START RECORDING
if doET
    % filename
    edf_fn = sprintf('OT%s%02d.edf',p.subName,p.currBlock);
    toRename_edf = fullfile(data_dir, sprintf('%s_S%s_B%02d.edf',expName,p.subName,p.currBlock));
    status = Eyelink('OpenFile', edf_fn);
    
    % timestamps of exp info
    % Eyelink('Message',sprintf('BLOCK_%02d_STARTED',b));
    Eyelink('Message', ['FILENAME: ', toRename_edf]);
    Eyelink('Message', ['SUBJECT: ', p.subName]);
    Eyelink('Message', ['BLOCK: ', sprintf('%02d', p.currBlock)]);
    Eyelink('Message', ['SYSTEM DATE AND TIME: ', ...
        datestr(now, 'dd-mm-yyyy HH:MM:SS')]);

    % Check which eye is available for gaze-contingent drawing. Returns 0 (left), 1 (right) or 2 (binocular)
    stim.eyeUsed = Eyelink('EyeAvailable');
    
    % START RECORDING
    Eyelink('StartRecording');
    Eyelink('Message','SESSION_STARTED');
end

%% Start Trial %% 
% gaze detection at each step, when violated fixation, give feedback and continue onto next trial
trial = 0; endExp = 0;
while 1 % while loop goes on forever until aborted manually 
    trial = trial + 1;
    if doET; Eyelink('Message',['TRIAL_', num2str(trial), '.START']); end %%% ET %%%
    %% LOAD IMAGES %%
    initLoad = GetSecs;
    % Target image
    targIdx = randi(360);
    imfn = fullfile(p.imagedir, sprintf('VCS_%d.jpg',targIdx));
    im = imread(imfn);
    im = changeBG(im,p.grey);
    targImText = Screen('MakeTexture',win,im);
    
    % MASK
    imfn = fullfile(p.imagedir, sprintf('VCS_%d.jpg',randi(360)));
    im = imread(imfn);
    im = changeBG(im,p.grey);
    % phase-scramble stimulus image for mask
    im_sc = phaseScrambleImage(im);
    maskImText=Screen('MakeTexture',win,im_sc);
    
    % DISTRACTOR
    for nd = 1:t.nDistFlip
        maskAngIdx = mod(targIdx+45+randi(360-90)-1,360)+1;
        imfn = fullfile(p.imagedir, sprintf('VCS_%d.jpg', maskAngIdx));
        im=imread(imfn);
        im = changeBG(im,p.grey);
        distImText(nd)=Screen('MakeTexture',win,im);
    end

    % determine Foveal or Peripheral for target
    % 0: both F and P, 1: only F, 2: only P
    if p.targCond == 0
        targLoc(trial) = randi(2); % stim location changes on each trial
    else
        targLoc(trial) = p.targCond; % only F if 1, only P if 2
    end
    tempRect = [0 0 p.stimImgSize(targLoc(trial)) p.stimImgSize(targLoc(trial))];
    
    % determine where for peripheral
    thisAng = randi(360); % randomly choose angular location
    stimRect = CenterRectOnPoint(tempRect, ...
        cx + cosd(thisAng)*p.stimEccentricity(targLoc(trial)),...
        cy - sind(thisAng)*p.stimEccentricity(targLoc(trial)));
    % assign distractor locations too
    distLoc(trial) = randi(3); % 1: fovea, 2: periphery, 3: no dist
    if distLoc(trial) < 3 % yes dist
        tempRect = [0 0 p.stimImgSize(distLoc(trial)) p.stimImgSize(distLoc(trial))];
        distRect = CenterRectOnPoint(tempRect, ...
            cx + cosd(thisAng)*p.stimEccentricity(distLoc(trial)),...
            cy - sind(thisAng)*p.stimEccentricity(distLoc(trial)));
    else % no dist
        distRect = [];
    end

    % give minimum of 2 seconds to load everything -- TODO: check if it
    % does take only 2 seconds across trials
    while GetSecs - initLoad < 2; end 
    % TODO: flicker fixation point to signal trial onset
    % use gaze coordination from this point to baseline subsequent coords
    Screen('FillOval', win, p.white, p.fixationDot);
    Screen(win, 'Flip');
    Screen('FillOval', win, p.black, p.fixationDot);
    Screen(win, 'Flip',0.15);
    Screen('FillOval', win, p.white, p.fixationDot);
    Screen(win, 'Flip',0.15);
    if doET % get eye position from the beginning of the trial to baseline
        evt = Eyelink('NewestFloatSample');
        while ~isstruct(evt)
            evt = Eyelink('NewestFloatSample'); % repeat until we get data -- TODO: check if this works as intended
        end
        % Save sample properties as variables. See EyeLink Programmers Guide manual > Data Structures > FSAMPLE
        stim.baseline_XY = [evt.gx(eyeUsed+1) evt.gy(eyeUsed+1)]; % [left eye gaze x, right eye gaze x] +1 as we're accessing a Matlab array
    end

    % pre-assign some values to variables
    p.respError(trial) = nan; % to differentiate when the actual error is 0

    %% SHOWING STIMULUS %%
    % show TARGET
    Screen('DrawTexture', win, targImText,[],stimRect);
    Screen('FillOval', win, p.black, p.fixationDot);
    initTarget = Screen(win, 'Flip');
    if doET; Eyelink('Message',['TRIAL_', num2str(trial), '.TARGET']); end %%% ET %%%
    while GetSecs - initTarget < t.stimDur - 0.001; stim = gazeControl_SP(doET,p,trial,stim,win); 
        if stim.gazeViolationIdx(trial) == 1; break; end % break while loop
    end
    if stim.gazeViolationIdx(trial) == 1; continue; end % go on to next trial

    % show mask
    Screen('DrawTexture', win, maskImText,[],stimRect);
    Screen('FillOval', win, p.black, p.fixationDot);
    initMask = Screen(win, 'Flip');
    if doET; Eyelink('Message',['TRIAL_', num2str(trial), '.MASK']); end %%% ET %%%
    while GetSecs - initMask < t.maskDur - 0.001; stim = gazeControl_SP(doET,p,trial,stim,win); 
        if stim.gazeViolationIdx(trial) == 1; break; end % break while loop
    end
    if stim.gazeViolationIdx(trial) == 1; continue; end % go on to next trial
    
    % delay
    if doET; Eyelink('Message',['TRIAL_', num2str(trial), '.DELAY']); end %%% ET %%%
    if distLoc(trial) < 3 % yes dist
        Screen('FillOval', win, p.black, p.fixationDot);
        initBlank = Screen(win, 'Flip');
        while GetSecs - initBlank < .5 - 0.001; stim = gazeControl_SP(doET,p,trial,stim,win); 
            if stim.gazeViolationIdx(trial) == 1; break; end % break while loop
        end
        if stim.gazeViolationIdx(trial) == 1; continue; end % go on to next trial
        for rep = 1:(t.delay-.5)
            % show distractor
            Screen('DrawTexture', win, distImText(rep),[],distRect);
            Screen('FillOval', win, p.black, p.fixationDot);
            initDist = Screen(win, 'Flip');
            while GetSecs - initDist < t.distDur - 0.001; stim = gazeControl_SP(doET,p,trial,stim,win); 
                if stim.gazeViolationIdx(trial) == 1; break; end % break while loop
            end
            if stim.gazeViolationIdx(trial) == 1; break; end % break rep for loop
            
            Screen('FillOval', win, p.black, p.fixationDot);
            initBlank = Screen(win, 'Flip');
            while GetSecs - initBlank < t.distDur - 0.001; stim = gazeControl_SP(doET,p,trial,stim,win); 
                if stim.gazeViolationIdx(trial) == 1; break; end % break while loop
            end
            if stim.gazeViolationIdx(trial) == 1; break; end % break rep for loop
        end
    elseif distLoc(trial) == 3 % no dist
        Screen('FillOval', win, p.black, p.fixationDot);
        initBlank = Screen(win, 'Flip');
        while GetSecs - initBlank < t.delay - 0.001; stim = gazeControl_SP(doET,p,trial,stim,win); 
            if stim.gazeViolationIdx(trial) == 1; break; end % break while loop
        end
    end
    if stim.gazeViolationIdx(trial) == 1; continue; end % go on to next trial
    
    %% RESPONSE %%
    % adjust shape on the ring to match the target
    % showing the first test stim
    tA = randi(360);
    imfn = fullfile(imgDir, sprintf('VCS_%d.jpg', tA));
    im=imread(imfn);
    im = changeBG(im,p.grey);
    testImg = Screen('MakeTexture',win,im);
    Screen('DrawTexture', win, testImg,[],stimRect);
    Screen('FillOval', win, p.white, p.fixationDot); % prompt for response
    initTest = Screen(win, 'Flip');
    if doET; Eyelink('Message',['TRIAL_', num2str(trial), '.RESP']); end %%% ET %%%
    
    % wait for response
    clear keyIsDown keyCode DPtime
    while GetSecs - initTest < t.respDur
        [keyIsDown, ~, keyCode] = KbCheck();
        if keyIsDown == 1
            % adjust coordinates based on pressed key
            if keyCode(p.keyBigLeft)==1
                tA = mod(tA-p.bigStep-1,360)+1;
            elseif keyCode(p.keySmallLeft)==1
                tA = mod(tA-p.smallStep-1,360)+1;
            elseif keyCode(p.keySmallRight)==1
                tA = mod(tA+p.smallStep-1,360)+1;
            elseif keyCode(p.keyBigRight)==1
                tA = mod(tA+p.bigStep-1,360)+1;
            elseif keyCode(p.keyEscape)==1
                if doET
                    Eyelink('Message','SESSION_ABORTED');
                    Eyelink('StopRecording');
                    Eyelink('CloseFile');
                    Eyelink('Command','clear_screen 0');
                end
                ShowCursor;
                ListenChar(0);
                Screen('CloseAll');

                return
            elseif keycode(p.keyEnd)==1
                endExp = 1;
                break
            else % if anything else if pressed
                keyIsDown = 0;
            end
            clear keyCode
            FlushEvents('keyDown');
        end
        % display updated image based on the new coordinates
        % make sure tA is within 1-360 range
        imfn = fullfile(imgDir, sprintf('VCS_%d.jpg', tA));
        im=imread(imfn);
        im = changeBG(im,p.grey);
        testImg = Screen('MakeTexture',win,im);
        Screen('DrawTexture', win, testImg,[],stimRect);
        Screen('FillOval', win, p.white, p.fixationDot);
        Screen(win, 'Flip');
    end

    % close image textures to free up memory
    Screen('Close',[targImText maskImText distImText testImg]);
%     Screen('Close', [windowOrTextureIndex or list of textureIndices/offscreenWindowIndices]);
    
    % record response
    p.respError(trial) = targIdx-tA;

    % if end key is pressed, break out of trial loop
    if endExp == 1
        break
    end
end

%% save data for this block
% save everything every block
p.targLoc = targLoc;
p.distLoc = distLoc;
save(fullfile(data_dir,baseName),'p','t','stim');

%%% ET %%%
if doET
    Eyelink('Message','SESSION_ENDED'); %%% ET %%%
    Eyelink('StopRecording');
    Eyelink('CloseFile');
    Eyelink('Command','clear_screen 0');
    
    % To transfer the .edf file to the stimulus presentation machine:
    % %% now try to receive the data file
    try
        fprintf('Receiving data file ''%s''\n', edf_fn, toRename_edf);
        status = Eyelink('ReceiveFile', edf_fn, toRename_edf);
        if status > 0
            fprintf('Received file! File size %d\n', status);
        else
            fprintf('NO EDF FILE RECEIVED.\n');
        end
        if exist(toRename_edf, 'file')==2
            fprintf('Data file ''%s'' can be found in ''%s''\n', toRename_edf);
        end
    catch
        fprintf('Problem receiving data file ''%s''\n', toRename_edf);
    end
end

%% end experiment

% show performance
text_inst = sprintf('Average error : %2.1f', ...
    round(mean(abs(p.respError(stim.gazeViolationIdx~=1))),1));
DrawFormattedText(win, text_inst, 'center', cy - text_size_pix*2, p.white);
Screen(win, 'Flip');
KbWait();
        
ShowCursor;
ListenChar(0);

Screen('CloseAll');

%% show performance

% 1. overall violations
fprintf('Total number of violations: %d(%3.1f%%)\n',sum(stim.gazeViolationIdx==1),...
    sum(stim.gazeViolationIdx==1)/length(stim.gazeViolationIdx)); % total # of violated trials (%)

% 2. violations for each target location
fprintf('Foveal: %d(%3.1f%%)\n',...
    sum(stim.gazeViolationIdx==1 && p.targLoc==1),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==1)/sum(stim.gazeViolationIdx==1));
fprintf('Peripheral: %d(%3.1f%%)\n',...
    sum(stim.gazeViolationIdx==1 && p.targLoc==2),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==2)/sum(stim.gazeViolationIdx==1));

% 3. violations for each target location X distractor condition
fprintf('Out of Foveal trials, Foveal distractor: %d(%3.1f%%), Peripheral distractor: %d(%3.1f%%), No distractor: %d(%3.1f%%)\n',...
    sum(stim.gazeViolationIdx==1 && p.targLoc==1 && p.distLoc == 1),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==1 && p.distLoc == 1)/sum(stim.gazeViolationIdx==1 && p.targLoc==1),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==1 && p.distLoc == 2)/sum(stim.gazeViolationIdx==1 && p.targLoc==1),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==1 && p.distLoc == 3)/sum(stim.gazeViolationIdx==1 && p.targLoc==1));
fprintf('Out of Peripheral trials, Foveal distractor: %d(%3.1f%%), Peripheral distractor: %d(%3.1f%%), No distractor: %d(%3.1f%%)\n',...
    sum(stim.gazeViolationIdx==1 && p.targLoc==2 && p.distLoc == 1),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==2 && p.distLoc == 1)/sum(stim.gazeViolationIdx==1 && p.targLoc==2),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==2 && p.distLoc == 2)/sum(stim.gazeViolationIdx==1 && p.targLoc==2),...
    sum(stim.gazeViolationIdx==1 && p.targLoc==2 && p.distLoc == 3)/sum(stim.gazeViolationIdx==1 && p.targLoc==2));

% plot performance
figure(1); clf; % scatter plot of errors with red markers as violated trials
subplot(2,1,1); 
scatter(find(stim.gazeViolationIdx ~= 1 && p.targLoc==1),p.respError(stim.gazeViolationIdx ~= 1 && p.targLoc==1),'k'); hold on;
scatter(find(stim.gazeViolationIdx == 1 && p.targLoc==1),zeros(1,sum(stim.gazeViolationIdx == 1 && p.targLoc==1)),'r');
subplot(2,1,2);
scatter(find(stim.gazeViolationIdx ~= 1 && p.targLoc==2),p.respError(stim.gazeViolationIdx ~= 1 && p.targLoc==2),'k'); hold on;
scatter(find(stim.gazeViolationIdx == 1 && p.targLoc==2),zeros(1,sum(stim.gazeViolationIdx == 1 && p.targLoc==2)),'r');

end

function imGray = changeBG(im,grey)
% change background from white to gray
maxval = max(double(im(:)));
imGray = im./maxval*grey;
end

function imScrambled = phaseScrambleImage(im)
% modified from imscramble.m by Martin Hebart (2009)
% http://martin-hebart.de/webpages/code/stimuli.html
p=1;
imclass = class(im); % get class of image
im = double(im);
imSize = size(im);
RandomPhase = p*angle(fft2(rand(imSize(1), imSize(2)))); %generate random phase structure in range p (between 0 and 1)
RandomPhase(1) = 0; % leave out the DC value

imFourier = fft2(im);     % Fast-Fourier transform
Amp = abs(imFourier);     % amplitude spectrum
Phase = angle(imFourier);   % phase spectrum
Phase = Phase + RandomPhase; % add random phase to original phase
% combine Amp and Phase then perform inverse Fourier
imScrambled = ifft2(Amp.*exp(sqrt(-1)*(Phase)));
imScrambled = real(imScrambled); % get rid of imaginery part in image (due to rounding error)

imScrambled = cast(imScrambled,imclass); % bring image back to original class
end

function [stim] = gazeControl_SP(doET,p,trial,stim,win)
% New version of eyeFeedback VER 17/3/2017
% - Shows differential feedback for blinks and saccades
% - instead of aborting a trial for any violation, a time window of
% violation (p.violationMaxDur) can be specified in the main experimental
% file. Decreases the number of false alarms
% - has fewer input variables to make handling easier

% edited from gazeControl_KA2.m
% only check for saccades and movement away from fixation
% doET: if 0, return without doing anything
% p: structure with experiment info
% t: trial index
% stim: data structure to record output
% win: window pointer

% TODO: detect violation & record violation duration
% show trajectory if violated and abort trial
% might need to create exception for blinks

% skip if not doing eyetracking
if doET == 0
    return
end

% Settings
saccTrajCol = [60 60 60]; % color for saccade trajectory displayed to subjects
dispTime = 1; % how long are error messages shown (in secs)

% Read in data
EyeLinkDefaults = p.EyeLinkDefaults;
error=Eyelink('CheckRecording');
if(error~=0)
    return;
end

% Get the sample in the form of an event structure
evt = Eyelink('NewestFloatSample'); % receive data from eye tracker

% Check if structure is present (removing this check might crash the program)
if ~isstruct(evt)
    stim.gazeViolationType(trial) = 404;
    stim.gazeViolationIdx(trial) = 1;
%     stim.evt_problem_tracker(trial) = 1;
else % if structure is there, proceed
    trackedEye = stim.eyeUsed+1; % predefined eye, 1 = left, 2 = right
    x = evt.gx(trackedEye); % get x coordinate from left and right eye
    y = evt.gy(trackedEye); % get y coordinate from left and right eye
    evtype=Eyelink('getnextdatatype');

    %checks for saccade violations
    if stim.gazeViolationIdx(trial) == 0 && evtype == EyeLinkDefaults.ENDSACC && p.saccDet
        % if feedback is on, show subject where saccade went
        if p.eyeDirectFB == 1
            Screen('DrawLine',win,saccTrajCol,p.xCenter,p.yCenter,max(x),max(y),3);
            Screen('FillOval', win, p.black, p.fixationDot);
            Screen('Flip', win);
            WaitSecs(dispTime);
        end
        stim.gazeViolationIdx(trial) = 1; % save info that violation occurred during this display
        stim.gazeViolationType(trial) = 200;
    end

    % if no saccade violations, check for fixation violations
    if stim.gazeViolationIdx(trial) == 0
        % if distance to fixation is within tolerance
        withinTol = sqrt((x-p.xCenter)^2+(y-p.yCenter)^2) <= p.fixTol;
        if sum(withinTol) == 0  % No good stim on any eye?
            if isnan(stim.violationOnset)
                stim.gazeViolationEvents(trial) = stim.gazeViolationEvents(trial) + 1;
                stim.violationOnset(trial) = GetSecs;
            else
                if GetSecs - stim.violationOnset(trial) > p.violationMaxDur
                    % if feedback is on, show subject where saccade went
                    if p.eyeDirectFB == 1
                        Screen('DrawLine',win,saccTrajCol,p.xCenter,p.yCenter,max(x),max(y),3);
                        Screen('FillOval', win, p.black, p.fixationDot);
                        Screen('Flip', win);
                        WaitSecs(dispTime);
                    end
                    stim.gazeViolationIdx(trial) = 1; % save info that violation occurred during this display
                    stim.gazeViolationType(trial) = 300;
                end
            end
        else
            % if there was no violation for this data point, reset timer
            stim.violationOnset(trial) = NaN;
        end
    end
end
end

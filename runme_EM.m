function runme_EM()
%-------------------------------------------------------------------------
% Script to run multiple experiment scripts in a row
%-------------------------------------------------------------------------
clear all;  % clear everything out!
close all;  % close existing figures
warning('off','MATLAB:dispatcher:InexactMatch');  % turn off the case mismatch warning (it's annoying)
dbstop if error  % tell us what the error is if there is one
AssertOpenGL;    % make sure openGL rendering is working (aka psychtoolbox is on the path)

%-------------------------------------------------------------------------
% save date and time!
%-------------------------------------------------------------------------
p.date_time_session = clock;
%-------------------------------------------------------------------------
% Important options for all experiments
%-------------------------------------------------------------------------
p.environment = 2; % 1 = Linux machine, 2 = iMac, 3 = PC
p.portCodes = 2;  %1 = use p.portCodes (we're in the booth), 0 is fMRI, 2 = laptop
p.windowed = 1; % 1 = small win for easy debugging!, otherwise 0
p.startClick = 1; % 1 = must press spacebar to initiate each trial.
p.shortTrial = 1; % 1 is short trials for debugging
p.debug = 0;
p.MRI = 0;
p.gammacorrect = false;
%-------------------------------------------------------------------------
% Build an output directory & check to make sure it doesn't already exist
%-------------------------------------------------------------------------
p.root = pwd;

addpath([p.root,'/SupportFunctions'])
if p.environment == 1
    if p.portCodes == 1
        p.datadir = '/mnt/pclexp/Documents/Holly/Data';
        p.GeneralUseScripts = '/mnt/pclexp/Holly/GeneralUseScripts';
    else
        p.datadir = '/Users/hkular/Documents/Github/EyeMove';
        p.GeneralUseScripts = '/Users/hkular/Documents/Github/GeneralUseScripts';
    end
    addpath(genpath(p.GeneralUseScripts));
    addpath(genpath([p.GeneralUseScripts,'/Calibration']));
else % just save locally!
    if ~exist([p.root, filesep,'Data',filesep], 'dir')
        mkdir([p.root, filesep,'Data',filesep]);
    end
    p.datadir = [p.root, filesep,'Data',filesep];
end

commandwindow();
%% ----------------------------------------------
 % collect subject info
 %-----------------------------------------------
    p.room = input('Room letter: ', 's'); if isempty(p.room); error('Enter the room letter!'); end
    practice = input('Practice? y = yes, n = no: ', 's'); if isempty(practice); error('Did the subject already complete a practice run?'); end
    info.Name = input('Subject initials: ', 's'); if isempty(info.Name); info.Name = 'tmp'; end % collect subject initials
    SubNum = input('Subject number: ', 's'); if isempty(SubNum); SubNum = '00'; end % collect subject number
    info.SubNum = sprintf('%02d', str2double(SubNum));
    % demographics
%     info.Age = input('Subject Age: ', 's'); if isempty(info.Age); info.Age = 'tmp'; end
%     info.Gender = input('Gender (M = male, F = female, O = non-binary or other, NR = prefer not to respond):' , 's'); if isempty(info.Gender); info.Gender = 'tmp'; end
%     info.Handed = input('Handedness (R = right, L = left, O = other, NR = prefer not to respond):', 's'); if isempty(info.Handed); info.Handed = 'tmp'; end
%     
%% -------------------------------------------------------------------------
% Run Main Experiment Scripts
%-------------------------------------------------------------------------

if practice == 'y'
    WM_MoveV1practice(p, info, 0, 1, 1);
end
WM_MoveV1(p, info, 0, 1, 1); % p info doET nruns startrun

%-------------------------------------------------------------------------
% Change back screen to default mode!
%-------------------------------------------------------------------------
if p.environment == 1
s = setScreen_Default();
if s == 0
    fprintf('Screen successfully set back to default!');
end
end
%-------------------------------------------------------------------------
% Postpare the environment
%-------------------------------------------------------------------------
ListenChar(0);
if p.environment == 3
    ShowHideWinTaskbarMex(1);
end
%% 
close all;
end


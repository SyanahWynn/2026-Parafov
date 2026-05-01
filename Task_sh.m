%% SET-UP %%
% Clear the workspace and the screen
close all;
clearvars;
% focus on the commandwindow
commandwindow

%% PARTICIPANT INFO %%
% https://ch.mathworks.com/help/matlab/ref/inputdlg.html
% https://ch.mathworks.com/matlabcentral/fileexchange/25862-inputsdlg-enhanced-input-dialog-box
% General setup
addpath('INPUTSDLG')
Title = 'Participant info'; % title of the GUI window
Options.Resize = 'off'; % Make dialog resizable: 'on' | {'off'}
Options.Interpreter = 'tex'; % Label text interpreter: 'latex' | {'tex'} | 'none'
Options.CancelButton = 'on'; % Show Cancel button: {'on'} | 'off'
Options.ApplyButton = 'on'; % Adds Apply button: 'on' | {'off'}
Options.ButtonNames = {'Continue','Cancel'}; % Customize OK|Cancel|Apply button names: {up to 3 elements}
Option.Dim = 2; % Horizontal dimension in fields
Prompt = {}; % Name of the input element <- [prompt string, output name, units, tooltip]
Formats = {}; % Format of the input element
DefAns = struct([]); % Default of the input element
% Experiment specific setup
Prompt(1,:) = {'Participant code', 'code',[]};
Formats(1,1).type = 'edit';
Formats(1,1).format = 'text';
DefAns(1).code = [num2str(yyyymmdd(datetime)) '#B'];
Prompt(end+1,:) = {'Participant number', 'ppn',[]};
Formats(1,2).type = 'edit';
Formats(1,2).format = 'integer';
Formats(1,2).limits = [101 160];
DefAns.ppn = 160;
Prompt(end+1,:) = {'Participant age', 'age',[]};
Formats(2,1).type = 'edit';
Formats(2,1).format = 'integer';
Formats(2,1).limits = [18 99];
DefAns.age = 99;
Prompt(end+1,:) = {'Participant gender', 'gender',[]};
Formats(2,2).type = 'edit';
Formats(2,2).format = 'text';
DefAns.gender = 'prefer not to say';
Prompt(end+1,:) = {'Experiment Details','exp',[]};
Formats(5,1).type = 'table';
Formats(5,1).format = {'char', 'logical'}; 
Formats(5,1).items = {'Option' 'Select'};
Formats(5,1).size = [182 112];
DefAns.exp = {'MEG' true 
              'EYE' true 
              'DEMO' false 
              'PRAC' true
              'LAB' true};
% Draw the gui
subj = inputsdlg(Prompt,Title,Formats,DefAns,Options);
% transform some of the info to the "var" variable
var = cell2struct(subj.exp(:,2),subj.exp(:,1),1);
subj = rmfield(subj,'exp');

% prevent chatacters to be typed in the matlab command window
ListenChar(2)

%% PSYCHTOOLBOX SETUP %%
% http://psychtoolbox.org/download.html#Windows
% synchronization of Psychtoolbox to the vertical retrace (VBL) is not working on this setup.
% You can force Psychtoolbox to continue, despite the severe problems, by adding the command
% Screen('Preference', 'SkipSyncTests', 1); at the top of your script, 
% if you really know what you are doing.
if ~var.LAB
    Screen('Preference', 'SkipSyncTests', 1)
end
% Change current directory
cd(fileparts(which('ParaFovMEG.m')))
% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);
% Get the screen numbers. This gives us a number for each of the screens
% attached to our computer.
% To draw we select the maximum of these numbers. So in a situation where we
% have two screens attached to our monitor we will draw to the external
% screen.
if var.LAB
    cfg.screenNumber = max(Screen('Screens'));
else
    cfg.screenNumber = 1;
end
% Determine the start time of the experiment
cfg.expStart                            = GetSecs;
% Define black and white (white will be 1 and black 0). This is because
% in general luminace values are defined between 0 and 1 with 255 steps in
% between. All values in Psychtoolbox are defined between 0 and 1.
% Do a simply calculation to calculate the luminance value for grey. This
% will be half the luminace values for white.
cfg.white                               = WhiteIndex(cfg.screenNumber);
cfg.black                               = BlackIndex(cfg.screenNumber);
cfg.grey                                = GrayIndex(cfg.screenNumber);
% Open an on screen window using PsychImaging and color it grey.
[cfg.window, cfg.windowRect]            = PsychImaging('OpenWindow', cfg.screenNumber, cfg.grey);
% Get the size of the on screen window
[cfg.screenXpixels, cfg.screenYpixels]  = Screen('WindowSize', cfg.window);
% Get the centre coordinate of the window in pixels
% For help see: help RectCenter
[cfg.xCenter, cfg.yCenter]              = RectCenter(cfg.windowRect);
% Query the frame duration
cfg.ifi                                 = Screen('GetFlipInterval', cfg.window);
% get the framerate (in Hz)
cfg.frameRate                           = Screen('NominalFrameRate', cfg.window);
% Retrieve the maximum priority number and set the PTB priority level to
% maximum. This means PTB will take processing priority over other system
% and applicaiton processes.
Priority(MaxPriority(window));
% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', cfg.window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
% hide cursor on experiment window
HideCursor(cfg.window)


%% VARIABLES %%
var.txtsz           = 30; % standard text size of the experiment
var.heightperc      = .15; % height of the image as a percentage of the screen
var.primeXpos       = [cfg.screenXpixels*.415 cfg.screenXpixels*.585]; % position on the x-axis of the parafoveal items
var.primeYpos       = cfg.screenYpixels*.5; % position on the y-axis of the parafoveal items
var.targetXpos      = cfg.screenXpixels*.5; % position on the x-axis of the target items
var.targetYpos      = cfg.screenYpixels*.5; % position on the y-axis of the target items
var.imgsLoc         = 'Pics'; % location of the images
var.imgLocPrac      = 'Pics/_practice'; % location of the practice images
var.subFileLoc      = 'SubjectFiles'; % location of the input files
var.subFileNameMain = '_ParaFov.mat'; % name of the specific input files
var.subFileNameMem  = '_Ret.mat'; % name of the specific input files
var.fixDur          = .100; % in seconds (jittered) (in the case of the memory task, fixation is presented longer if the participant is still holding down a key.
var.primeDur        = .150; % in seconds
var.targetDur       = .500; % in seconds
var.retDur          = 1; % in seconds
% I have now split up the ITI fixation in two parts (one at the start of the
% trial and one at the end, to account for any timing issues in loading the
% stimuli
var.ITIDur          = [.900 .950 1 1.050 1.100]; % in seconds (jittered). 
var.numImages       = 2; % number of images on each prime screen
var.numPrimes       = 2; % number of primes per screen
var.numPriScrn      = 2; % number of prime screens
var.dataLoc         = 'Data'; % location where the data will be stored
% if the folder does not exist, make it.
if ~isfolder(var.dataLoc)
   mkdir(var.dataLoc)
end
% IMPORT %
% Get the subject files for the participant and if it is not found abort
SF                  = dir(var.subFileLoc);
% one for the main task and one for the memory task, two in total
if sum(contains({SF.name},num2str(subj.ppn)))==2
else
    % Clear the screen.
    sca;
end
clear SF
subj.fileinMain     = struct2cell(load([var.subFileLoc filesep num2str(subj.ppn) var.subFileNameMain])); subj.fileinMain=subj.fileinMain{1};
subj.fileinMem      = struct2cell(load([var.subFileLoc filesep num2str(subj.ppn) var.subFileNameMem])); subj.fileinMem=subj.fileinMem{1};
var.pracTrials      = 15; % number of practice trials
var.mainTrials      = size(subj.fileinMain,1); % total number of main task trials
var.blocks          = max(unique(subj.fileinMain.blockN));  % number of blocks
var.blockTrials     = var.mainTrials/var.blocks; % number of trials per block
var.memTrials       = size(subj.fileinMem,1); % total number of memory trials
% Set up the key coding
var.keyExit         = KbName('ESCAPE'); % key to abort the experiment (escape)
var.keyBreak        = KbName('SPACE'); % key to pause the experiment (space)
var.keyContinue     = KbName('RETURN'); % key to pause the experiment (space)
if mod(str2double(subj.ppn),2)==1
    var.keyOld      = [KbName('LEFTARROW') KbName('7&')]; % key to indicate old memory (left arrow/index finger)
    var.keyGuess    = [KbName('DOWNARROW') KbName('8*')]; % key to indicate new item (down arrow/middle finger)
    var.keyNew      = [KbName('RIGHTARROW') KbName('9(')]; % key to indicate new item (down arrow/middle finger)
    subj.memtxt     = '1 = old  2 = guess  3 = new';
else
    var.keyNew      = [KbName('LEFTARROW') KbName('7&')]; % key to indicate new memory (left arrow/index finger)
    var.keyGuess    = [KbName('DOWNARROW') KbName('8*')]; % key to indicate old item (down arrow/middle finger)
    var.keyOld      = [KbName('RIGHTARROW') KbName('9(')]; % key to indicate new item (down arrow/middle finger)
    subj.memtxt     = '1 = new  2 = guess  3 = old';
end
% set up the markers/triggers
% first fixation onset (start trial) (252), intermediate fixation onset (248)
var.trg.fixation    = [252 248]; % 11111100 & 11111000
% start subtasks: parafoveal (240), memory (224)
var.trg.task        = [240 224]; % 11110000 & 11100000
% the triggers below are set in the inputfile script
% parafov onset: first screen: no prime (1), prime 1-back (129), prime 2-back (65)
%                second screen:no prime (2), prime 1-back (130), prime 2-back (66) 
var.trg.pfov        = [1 129 65;... % 00000001, 10000001, 01000001
                       2 130 66];   % 00000010, 10000010, 01000010
% foveal onset: animal (192), clothing (160), food (144), plants (136), vehicles (132)
var.trg.fov         = [192 160 144 136 132]; % 11000000, 10100000, 10010000, 10001000, 10000100
% memory item onset: old (24), new (36)
var.trg.mem         = [24, 36]; % 00011000 & 00100100
% memory item response: hit (128), correct rejection (64), miss (32), false alarm(16), 
%                       guess (old) (8), guess (new) (4)
var.trg.resp        = [128, 64, 32, 16,... % 10000000, 01000000, 00100000, 00010000
                        8, 4]; % 00001000, 00000100

%% EYELINK %%
if var.EYE
    % set up for the eyelink
    var.el.eye          = 'LEFT'; % Eye used
    var.el.edfname      = [num2str(subj.ppn) '.edf']; % EDF filename
    var.el.edffolder    = [var.dataLoc filesep 'Eyelink' filesep]; % EDF filefolder
    % Add eyelink script folder (should be in main experiment folder)
    addpath('Eyelink');
    % determine whether to do calibration and or drift correction
    var.el.setup        = true;
    var.el.cali         = true;
    var.el.drift        = true;
    % Run the Eyelink function
    var.el.el = PF_EyelinkSetup(cfg,var.el);
end

%% BUTTON BOX %%
% initialisation sendTrigger
if var.MEG == 1
    % trigger (marker) can be integer from 1 to 255 (8 trigger channels)
    cfg.portAddress = hex2dec('BFF8');
    cfg.ioObjTrig = io64;
    status = io64(cfg.ioObjTrig);
    io64(cfg.ioObjTrig,cfg.portAddress,0); % trigger 0 (reset)
else
    cfg.portAddress = NaN;
    cfg.ioObjTrig = NaN;
end

%% EXPORT %%
% set data folder and Create path and name for the results file
var.dataFolder      = fullfile(var.dataLoc, '\');
subj.fileoutMain    = [var.dataFolder 'ParaFov_' num2str(subj.ppn) '_' regexprep(num2str(floor(clock)),' +','-')];
subj.fileoutMem     = [var.dataFolder 'Ret_' num2str(subj.ppn) '_' regexprep(num2str(floor(clock)),' +','-')];
% Preallocate data tables
Data_ParaFov        = table('Size',[var.mainTrials 21], 'VariableTypes',...
                    [cellstr("double"); cellstr(repmat("cell",2,1)); cellstr(repmat("double",3,1)); cellstr(repmat("cell",11,1)); cellstr(repmat("double",4,1))],'VariableNames',...
                    {'ppn' 'code' 'gender' 'age' 'trial' 'block' 'cond'...
                    'targetName' 'targetCat' 'prime1Name' 'prime1Cat' 'prime2Name' 'prime2Cat'...
                    'prime3Name' 'prime3Cat' 'prime4Name' 'prime4Cat' 'timeFix1' 'timePfov1' 'timePfov2' 'timeFov'}); % 'timeFix1' onset of first fixation of the trial
Data_Retrieval      = table('Size',[var.memTrials 15], 'VariableTypes',...
                    [cellstr("double"); cellstr(repmat("cell",2,1)); cellstr(repmat("double",3,1)); cellstr(repmat("cell",4,1)); cellstr(repmat("double",5,1))],'VariableNames',...
                    {'ppn' 'code' 'gender' 'age' 'trial' 'block' 'cond' 'type'...
                    'itemName' 'itemCat' 'resp' 'memcat' 'RT' 'timeFix' 'timeItem'});

%% DEMO VERSION %%
% make a short version of the task (10% of trials per block) to check things
if var.DEMO
    subj.fileinMain     = subj.fileinMain(1:var.blockTrials*.10:var.mainTrials,:);
    subj.fileinMem      = subj.fileinMem(1:var.memTrials/var.blocks*.10:var.memTrials,:);
    var.mainTrials      = size(subj.fileinMain,1); % total number of main task trials
    var.blockTrials     = var.mainTrials/var.blocks; % number of trials per block
    var.memTrials       = size(subj.fileinMem,1); % total number of memory trials  
end                
                
%% TEXT: INTRO %%
PF_text(['As a reminder: \n\nIn part A, you will see images on the screen, try to remember the images you see in the centre of the screen. \nAfter a short break, part B will start and you will be asked to indicate which of the images you saw in part A.\n' subj.memtxt '\n\nIn a moment you will start practicing the task with a few examples. \n\n\nPress a button to continue.'],...
    var.txtsz,'center','center',cfg.window,cfg.black);
WaitSecs(2);
KbStrokeWait;

%% PRACTICE %%
% loop over the practice trials
if var.PRAC
    pr=1;
else
    pr=16;
end
while pr<=var.pracTrials
    if pr ==1
        % make the practice trials
        [prc.fileinMain, prc.fileinMem] = PF_pracList(var.pracTrials, var.imgLocPrac);
    end
    % FIXATION
    PF_text('+',...
        var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
    WaitSecs(var.ITIDur(randi(length(var.ITIDur),1,1)));
    for p=1:var.numPrimes*var.numPriScrn
        [prc.(['prime' num2str(p) 'Img']), prc.(['prime' num2str(p) 'Rect'])] =...
            PF_image(prc.fileinMain.(['pfov' num2str(p) '_img']){pr},var.imgLocPrac,...
            cfg.window,cfg.screenYpixels.*var.heightperc,var.primeXpos([mod(p,2)==1,mod(p,2)==0]), var.primeYpos);
    end
    [prc.targetImg, prc.targetRect] =...
        PF_image(prc.fileinMain.fov_img{pr},var.imgLocPrac,...
        cfg.window,cfg.screenYpixels.*var.heightperc,var.targetXpos, var.targetYpos);
    for p=1:var.numPrimes  
        % PRIMES 
        Screen('DrawTexture', cfg.window, prc.(['prime' num2str(2*p-1) 'Img']), [], prc.(['prime' num2str(2*p-1) 'Rect']));
        Screen('DrawTexture', cfg.window, prc.(['prime' num2str(2*p) 'Img']), [], prc.(['prime' num2str(2*p) 'Rect']));       
        PF_text('+',...
            var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
        WaitSecs(var.primeDur);
        % FIXATION
        PF_text('+',...
            var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
        WaitSecs(var.fixDur(randi(length(var.fixDur),1,1)));
    end
    % TARGET
    Screen('DrawTexture', cfg.window, prc.targetImg, [], prc.targetRect);
    Screen('Flip', cfg.window); %fix this, frame rate stuff!
    % Wait for duration of target time
    WaitSecs(var.targetDur);
    if pr==var.pracTrials
        % BREAK
        prc.t1 = GetSecs;
        while(GetSecs-prc.t1<=6)
            PF_text(['Short break for ' num2str(floor(6-(GetSecs-prc.t1))) ' seconds before we continue'],...
                var.txtsz,'center','center',cfg.window,cfg.black);
        end
        for m=1:var.pracTrials*2/3
            % FIXATION
            PF_text('+',...
                var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
            [keyIsDown,trlm.t2, keyCode] = KbCheck;
            while keyIsDown
                [keyIsDown,trlm.t2, keyCode] = KbCheck;
            end
            WaitSecs(var.ITIDur(randi(length(var.ITIDur),1,1))/2);
            % RETRIEVAL
            [prcm.itemImg, prcm.itemRect] =...
                PF_image(prc.fileinMem.item{m},var.imgLocPrac,...
                cfg.window,cfg.screenYpixels.*var.heightperc,var.targetXpos, var.targetYpos);
            Screen('DrawTexture', cfg.window, prcm.itemImg, [], prcm.itemRect);
            PF_text(subj.memtxt,...
                var.txtsz*(2/3),'center',cfg.screenYpixels*.70,cfg.window,cfg.black);
            prcm.t1 = GetSecs;
            resp = false;
            while ~resp && (GetSecs-prcm.t1<5)
                [~,~, keyCode] = KbCheck;
                trlm.resp = find(keyCode);
                if ismember(find(keyCode),[var.keyOld var.keyGuess var.keyNew])
                    resp = true;
                    if strcmp(prc.fileinMem.cat{m},'fov') && ismember(trlm.resp,var.keyOld) % hit
                        PF_text('correct!',...
                            var.txtsz*(1+1/3),'center','center',cfg.window,[0,255,0]);
                        WaitSecs(.5);
                    elseif strcmp(prc.fileinMem.cat{m},'new') && ismember(trlm.resp,var.keyNew) % correct rejection
                        PF_text('correct!',...
                            var.txtsz*(1+1/3),'center','center',cfg.window,[0,255,0]);
                        WaitSecs(.5);
                    elseif strcmp(prc.fileinMem.cat{m},'fov') && ismember(trlm.resp,var.keyNew) % miss
                        PF_text('incorrect!',...
                            var.txtsz*(1+1/3),'center','center',cfg.window,[255,0,0]);
                        WaitSecs(.5);
                    elseif strcmp(prc.fileinMem.cat{m},'new') && ismember(trlm.resp,var.keyOld) % false alarm
                        PF_text('incorrect!',...
                            var.txtsz*(1+1/3),'center','center',cfg.window,[255,0,0]);
                        WaitSecs(.5);
                    elseif ismember(trlm.resp,var.keyGuess) % guess (old)
                    end
                end
            end
            %fprintf(['\nResponse: ',num2str(find(keyCode))])
            while GetSecs-prcm.t1<=var.retDur
            end
            % END TRIAL
            Screen('Close')
            clear prcm
        end
        % END PRACTICE
        Screen('Close')
        clear prc
        prc.q = Ask(cfg.window,'Tell the researcher if you are ready to continue. Researcher, press n to practice again.',[],[cfg.grey],'GetChar', 'center');
        % researcher, press 'y' for continue, 'n' for practice again.
        if any(ismember(prc.q,'n'))
            pr=0;
            PF_text(['As a reminder: \n\nIn part A, you will see images on the screen, try to remember the images you see in the centre of the screen. \nAfter a few seconds, part B will start and you will be asked to indicate which of the images you saw in part A.\n' subj.memtxt '\n\nIn a moment you will start practicing the task with a few examples. \n\n\nPress a button to continue.'],...
                var.txtsz,'center','center',cfg.window,cfg.black);
            WaitSecs(2);
            KbStrokeWait;
        end   
    end
    pr=pr+1;
end

%% TEXT: START RECORDING %%
if var.EYE == 1
    % tell Eyelink the experiment has started
    Eyelink('message', 'SYNCTIME');
end
% display a reminder of the recording until the "continue" key is pressed
resp = false;
PF_text('Researcher, start the recording',...
        var.txtsz,'center','center',cfg.window,cfg.black);
while ~resp
    [~,~,keyCode] = KbCheck;
    if find(keyCode) == var.keyContinue
        resp = true;
    end
end

%% TEXT: START EXPERIMENT %%
PF_text(['As a reminder: \n\nIn part A, you will see images on the screen, try to remember the images you see in the centre of the screen. \nAfter a few seconds, part B will start and you will be asked to indicate which of the images you saw in part A.\n Press ' subj.memtxt '\n\n\nPress a button to start the experiment.'],...
    var.txtsz,'center','center',cfg.window,cfg.black);
WaitSecs(2);
KbStrokeWait;
% Send markers
trigger = var.trg.task(1); % start parafoveal task onset
sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);

%% FIXATION %%
% Draw fixation in the middle of the screen in Calibri in white
var.timeFix1 = PF_text('+',...
    var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
% Send markers
trigger = var.trg.fixation(1); % first fixation onset
sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
% Wait for twice the duration of fixation time to get the participants ready
WaitSecs(var.ITIDur(randi(length(var.ITIDur),1,1))*2);

%% MAIN EXPERIMENTAL LOOP %%
% Loop through all the rows from the file
m=1; % start of the memory count
for r=1:size(subj.fileinMain,1)     
    %% SET-UP %%
    %fprintf(['\nParaFov trlnr: ' num2str(r)])
    % loop over the primes
    for p=1:var.numPrimes*var.numPriScrn
        % Here we load in the primes from file from the current row. 
        % pfov1 = first screen left, pfov3 = second screen left,
        % pfov2 = first screen right, pfov4 = second screen right
        trl.(['prime' num2str(p) 'Name'])   = subj.fileinMain.(['pfov' num2str(p) '_img']){r};
        trl.(['prime' num2str(p) 'Cat'])    = subj.fileinMain.(['pfov' num2str(p) '_cat']){r};
        trl.(['prime' num2str(p) 'Dir'])    = [var.imgsLoc filesep trl.(['prime' num2str(p) 'Cat'])];
        [trl.(['prime' num2str(p) 'Img']), trl.(['prime' num2str(p) 'Rect'])] =...
            PF_image(trl.(['prime' num2str(p) 'Name']),trl.(['prime' num2str(p) 'Dir']),...
            cfg.window,cfg.screenYpixels.*var.heightperc,var.primeXpos([mod(p,2)==1,mod(p,2)==0]), var.primeYpos);
    end
    % Here we load in the target from file from the current row.
    trl.targetName   = subj.fileinMain.fov_img{r};
    trl.targetCat    = subj.fileinMain.fov_cat{r};
    trl.targetDir    = [var.imgsLoc filesep trl.targetCat];
    [trl.targetImg, trl.targetRect] =...
        PF_image(trl.targetName,trl.targetDir,...
        cfg.window,cfg.screenYpixels.*var.heightperc,var.targetXpos, var.targetYpos);

    %% FIXATION %%
    % Draw fixation in the middle of the screen in Calibri in white
    PF_text('+',...
        var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
    % Secretly not the first fixation, so we are not sending triggers now.
    % Wait for half the duration of fixation time
    WaitSecs(var.ITIDur(randi(length(var.ITIDur),1,1))/2);
    % check for break or exit
    PF_stop(var.keyBreak,var.keyExit)
        
    % Loop over the "prime/parafoveal" screens
    for p=1:var.numPriScrn  
        %% PRIMES %%
        % Draw the image to the screen, unless otherwise specified PTB will draw
        % the texture full size in the center of the screen. 
        Screen('DrawTexture', cfg.window, trl.(['prime' num2str(2*p-1) 'Img']), [], trl.(['prime' num2str(2*p-1) 'Rect']));
        Screen('DrawTexture', cfg.window, trl.(['prime' num2str(2*p) 'Img']), [], trl.(['prime' num2str(2*p) 'Rect']));       
        % Draw fixation in the middle of the screen in Calibri in white
        trl.(['timePfov' num2str(p)]) = PF_text('+',...
            var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
        % Send trigger
        trigger = subj.fileinMain.(['trg_pfov' num2str(p)])(r); % Parafoveal images onset
        sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
        % Wait for duration of prime time
        WaitSecs(var.primeDur);
        %% FIXATION %%
        % Draw fixation in the middle of the screen in Calibri in white
        PF_text('+',...
            var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
        % Send trigger
        trigger = var.trg.fixation(2); % fixation onset
        sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
        % Wait for duration of fixation time
        WaitSecs(var.fixDur(randi(length(var.fixDur),1,1)));
    end

    %% TARGET
    % Draw the image to the screen, unless otherwise specified PTB will draw
    % the texture full size in the center of the screen. 
    Screen('DrawTexture', cfg.window, trl.targetImg, [], trl.targetRect);
    % Flip to the screen
    trl.timeFov = Screen('Flip', cfg.window); 
    % Send trigger
    trigger = subj.fileinMain.trg_fov(r); % Foveal image onset 
    sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
    % Wait for duration of target time
    WaitSecs(var.targetDur);
    % clear the screen
    Screen('Close')
    
    %% FIXATION %%
    % Draw fixation in the middle of the screen in Calibri in white
    trl.timeFix = PF_text('+',...
        var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
    % Send trigger
    trigger = var.trg.fixation(1); % fixation onset
    sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
    % Wait for half the duration of fixation time
    WaitSecs(var.ITIDur(randi(length(var.ITIDur),1,1))/2);
    % check for break or exit
    PF_stop(var.keyBreak,var.keyExit)
    
    %% OUTPUT %%
    % Create results file
    Data_ParaFov(r,:) = [subj.ppn subj.code subj.gender subj.age r subj.fileinMain.blockN(r) subj.fileinMain.cond(r) ....
        trl.targetName trl.targetCat trl.prime1Name trl.prime1Cat trl.prime2Name trl.prime2Cat ...
        trl.prime3Name trl.prime3Cat trl.prime4Name trl.prime4Cat trl.timeFix-cfg.expStart trl.timePfov1-cfg.expStart trl.timePfov2-cfg.expStart trl.timeFov-cfg.expStart];
    
    % Check if it is break time
    if ismember(r,linspace(var.blockTrials,var.mainTrials,var.blocks))
        %% FIXATION %%
        % Draw fixation in the middle of the screen in Calibri in white
        PF_text('+',...
            var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
%         % Send markers
%         trigger = 99 % fixation onset
%         sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
        % Wait for duration of fixation time
        WaitSecs(var.ITIDur(randi(length(var.ITIDur),1,1))/2);
        
        %% BREAK %%
        % start of the 10 second break 
        trl.t1 = GetSecs;
        while(GetSecs-trl.t1<=11)
            PF_text([num2str(floor(11-(GetSecs-trl.t1))) ' seconds before we continue'],...
                var.txtsz,'center','center',cfg.window,cfg.black);
        end
        % Send markers
        trigger = var.trg.task(2); % fixation onset
        sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
        
        %% MEMORY EXPERIMENTAL LOOP %%
        % start the memory task at the end of each block after a short break
        while subj.fileinMain.blockN(r)==subj.fileinMem.blockN(m)
            %fprintf(['\nRetrieval trlnr: ' num2str(m)])
            %% FIXATION %%
            % Draw fixation in the middle of the screen in Calibri in white
            trlm.timeFix = PF_text('+',...
                var.txtsz*(1+1/3),'center','center',cfg.window,cfg.black);
            % Send trigger
            trigger = var.trg.fixation(1); % fixation onset
            sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
            % wait until all keys are released and then continue
            [keyIsDown,trlm.t2, keyCode] = KbCheck;
            while keyIsDown
                [keyIsDown,trlm.t2, keyCode] = KbCheck;
            end
            % Wait for duration of half the ITI fixation time
            WaitSecs(var.ITIDur(randi(length(var.ITIDur),1,1))/2);
            % check for break or exit
            PF_stop(var.keyBreak,var.keyExit)
            
            %% RETRIEVAL %%
            % Here we load in the target from file from the current row.
            trlm.itemName   = subj.fileinMem.item{m};
            trlm.itemCat    = subj.fileinMem.cat{m};
            trlm.itemDir    = [var.imgsLoc filesep trlm.itemCat];
            [trlm.itemImg, trlm.itemRect] =...
                PF_image(trlm.itemName,trlm.itemDir,...
                cfg.window,cfg.screenYpixels.*var.heightperc,var.targetXpos, var.targetYpos);
            % Draw the image to the screen, unless otherwise specified PTB will draw
            % the texture full size in the center of the screen. 
            Screen('DrawTexture', cfg.window, trlm.itemImg, [], trlm.itemRect);
            % Add the response options to the bottom of the screen
            trlm.timeItem = PF_text(subj.memtxt,...
                var.txtsz*(2/3),'center',cfg.screenYpixels*.70,cfg.window,cfg.black);
            % Send trigger
            trigger = subj.fileinMem.trg_mem(m); % Stimulus onset 
            sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
            % Wait for a response (max 5 sec)
            trlm.t1 = GetSecs;
            resp = false;
            while ~resp && (GetSecs-trlm.t1<5)
                % Check the keyboard
                [~,trlm.t2, keyCode] = KbCheck;
                trlm.resp = find(keyCode);
                if ismember(trlm.resp,[var.keyOld var.keyGuess var.keyNew])
                    if subj.fileinMem.trg_mem(m) == 24 && ismember(trlm.resp,var.keyOld) % hit
                        trigger = var.trg.resp(1);
                    elseif subj.fileinMem.trg_mem(m) == 36 && ismember(trlm.resp,var.keyNew) % correct rejection
                        trigger = var.trg.resp(2);
                    elseif subj.fileinMem.trg_mem(m) == 24 && ismember(trlm.resp,var.keyNew) % miss
                        trigger = var.trg.resp(3);
                    elseif subj.fileinMem.trg_mem(m) == 36 && ismember(trlm.resp,var.keyOld) % false alarm
                        trigger = var.trg.resp(4);
                    elseif subj.fileinMem.trg_mem(m) == 24 && ismember(trlm.resp,var.keyGuess) % guess (old)
                        trigger = var.trg.resp(5);
                    elseif subj.fileinMem.trg_mem(m) == 36 && ismember(trlm.resp,var.keyGuess) % guess (new)
                        trigger = var.trg.resp(6);
                    end
                    % Send trigger
                    sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
                    % set the response
                    trlm.memcat = trigger; % response
                    % end the while loop
                    resp = true;
                end
            end
            if ~resp
                % set the RT as NAN
                trlm.t2 = NaN;
                trigger = NaN;
                trlm.memcat = NaN;
                trlm.resp = NaN;
            end
            % if the response was made within var.retDur, keep presenting
            % the item for later MEG analyses
            while GetSecs-trlm.t1<=var.retDur
            end
            % clear the screen
            Screen('Close')
            
            %% OUTPUT %%
            % Create results file
            trlm.RT   = trlm.t2-trlm.t1;
            Data_Retrieval(m,:) = [subj.ppn subj.code subj.gender subj.age m subj.fileinMem.blockN(m) subj.fileinMem.cond(m) subj.fileinMem.type(m)....
                trlm.itemName trlm.itemCat trlm.resp trlm.memcat trlm.RT trlm.timeFix-cfg.expStart trlm.timeItem-cfg.expStart];
    
            %% END TRIAL %%
            % Add to the memory trial counter if the maximaum trial number
            % is reached, break out of the loop
            m=m+1;
            if m>var.memTrials
                break
            end
            clear trlm
        end
        % calculate memory performance in this block
        mem_acc_bl = sum(ismember(Data_Retrieval.memcat(Data_Retrieval.block==subj.fileinMem.blockN(m-1)),[128, 64]))/sum(ismember(Data_Retrieval.memcat(Data_Retrieval.block==subj.fileinMem.blockN(m-1)),[128, 64,32,16]));

        % save the data (as a back-up in case matlab crashes halfway)
        % If it does, the fixation column still needs to be altered (see
        % the saving below).
        save(subj.fileoutMain, 'Data_ParaFov')
        save(subj.fileoutMem, 'Data_Retrieval')
        
        %% BREAK %%
        % start of the max half a min break
        if m<=var.memTrials
            trl.t1 = GetSecs;
            resp = false;
            frst=1;
            while(GetSecs-trl.t1<=31) && ~resp
                PF_text(['You had ' num2str(mem_acc_bl*100) '% of the memory trials correct this block.' ...
                    '\n\nShort break for max ' num2str(floor(31-(GetSecs-trl.t1))) ' seconds before block the next block begins. \n\nOr press any key to continue'],...
                    var.txtsz,'center','center',cfg.window,cfg.black);
                % Wait for a bit on the first loop to let people actually read the text
                if frst
                    WaitSecs(2);
                end
                % Check if a response has been made
                [keyIsDown,~,~] = KbCheck;
                if keyIsDown
                    resp = 1;
                end
                frst = false;
            end
        end
        %% EYELINK/MEG: DRIFT CORRECTION / STOP -> START RECORDING %%
        if var.EYE
            % determine whether to do calibration and or drift correction
            var.el.setup        = false;
            var.el.cali         = false;
            var.el.drift        = true;
            % Run the Eyelink function
            PF_EyelinkSetup(cfg,var.el)
        end
%         % display a reminder of the recording until the "continue" key is pressed
%         resp = false;
%         PF_text('Researcher, continue the experiment.',...
%                 var.txtsz,'center','center',cfg.window,cfg.black);
%         while ~resp
%             [~,~,keyCode] = KbCheck;
%             if find(keyCode) == var.keyContinue
%                 resp = true;
%             end
%         end
        if r < size(subj.fileinMain,1)
            % Send markers
            trigger = var.trg.task(1); % start parafoveal task onset
            sendTrigger(cfg.ioObjTrig,cfg.portAddress,trigger,var.MEG,var.EYE);
        end
    end
    
    %% END TRIAL %%
    clear trl
end

%% END %%
% Show end text
PF_text('The task is now completed. Thank you! \nPlease wait until the researcher enters the room.',...
    var.txtsz,'center','center',cfg.window,cfg.black);
WaitSecs(3);

%% CLEAN-UP %%
% Adjust the fixation column. Since the fixation of the next trial starts
% in the loop of the previous one (to make sure the stimulus doesn't stay
% on screen for longer due to the end/start trial processes), the fixations
% are shifted in the output. Here we correct for that.
Data_ParaFov.timeFix1(2:end)    = Data_ParaFov.timeFix1(1:end-1);
Data_ParaFov.timeFix1(1)        = var.timeFix1-cfg.expStart;
% save the data
save(subj.fileoutMain, 'Data_ParaFov')
writetable(Data_ParaFov,[subj.fileoutMain '.csv'])
save(subj.fileoutMem, 'Data_Retrieval')
writetable(Data_Retrieval,[subj.fileoutMem '.csv'])
% close the Eyelink
if var.EYE
    PF_EyelinkClose(var.el)
end
% Clear the screen.
sca;
clearvars
% enable chatacters to be typed in the matlab command window
ListenChar(2)

%% LOCAL FUNCTIONS %%
% These could be rewritten to have cfg input, so if you are bored..
function SOT = PF_text(text,size,locx,locy,window,color)
    % This function displays the text on the screen, specified by the input
    Screen('TextSize', window, size);
    Screen('TextFont', window, 'Calibri');
    DrawFormattedText(window,...
        text,...
        locx, locy, color);
    % Flip to the screen and get an estimate of stimulus-onset time.
    [~,SOT,~,~] = Screen('Flip', window);
end
function [image,imageRectangle] = PF_image(name,imageDir,window,imageHeight,locx,locy)
    % get the details of the image to be presented. This function will
    % not present the image.
    % load the image
    image = imread([imageDir filesep name]);
    % Make the image into a texture
    image = Screen('MakeTexture', window, image);
    % Get the aspect ratio of the image. Assumes that both images are equally sized
    % We need this to maintain the aspect ratio of the image when we draw it 
    % different sizes. Otherwise, if we don't match the aspect ratio the image 
    % will appear warped / stretched We will set the height of each drawn image 
    % to a fraction of the screens height
    imageWidth = imageHeight .* (size(image,2) / size(image,1)); % adjusted height * aspect ratio (of the left image)
    % Make the destination rectangles for our image.
    imageRectangle = [0 0 imageWidth imageHeight];
    imageRectangle = CenterRectOnPointd(imageRectangle, locx, locy);
end
function PF_stop(keyBreak,keyExit)
    % Check if the break key is pressed indicating the need for a break
    % until a key is pressed again. Or if the exit key is pressed
    % indicating experiment abortion.
    [~, ~, keyCode] = KbCheck;
    if keyCode(keyBreak)
        WaitSecs(2);
        KbStrokeWait;
    elseif keyCode(keyExit)
        % Clear the screen.
        sca;
    end
end
function [T,M] = PF_pracList(N,imageDir)
    % make a very basic (random) list of practice items
    % turn off this warning
    warning('off','MATLAB:table:RowsAddedExistingVars')
    % make empty tables
    T = table;
    M = table;
    % get all the potential practice images
    images = dir(imageDir);
    images = {images(3:end).name};
    % shuffle
    images = images(randperm(length(images)));
    % Get the list for the main ParaFov task
    for p=1:N
        % select the foveal and parafoveal items
        T.fov_img(p) = images(1);
        pfov = images(1:5);
        pfov = pfov(randperm(length(pfov)));
        T.pfov1_img(p) = pfov(1);
        T.pfov2_img(p) = pfov(2);
        T.pfov3_img(p) = pfov(3);
        T.pfov4_img(p) = pfov(4);
        % clear those images from the list
        images(1:5)=[];
        clear pfov
    end
    % save all the possible items for the memory task
    items.fov = T.fov_img; % foveal
    %items.pfov = reshape(table2cell(T(:,2:end)),numel(T(:,2:end)),1); % parafoveal
    items.new = images'; % new
    % and shuffle
    items.fov = items.fov(randperm(length(items.fov))); % foveal
    % items.pfov = items.pfov(randperm(length(items.pfov))); % parafoveal
    items.new = items.new(randperm(length(items.new))); % new
    % Get the list for the memory task 
    cats = fieldnames(items);
    for m=1:floor(N*2/3)
        % randomly choose from a foveal or new image
        c = randi(length(cats));
        M.item(m) = items.(cats{c})(1);
        M.cat(m) = cats(c);
        items.(cats{c})(1) = [];
    end
end
function sendTrigger(ioObjTrig,portAddress,trigger,MEG,EYE)
    % Function to send MEG and eyelink triggers, originally made by Yali Pan. Adjusted version is used here.
    if MEG
        % Send trigger to MEG
        io64(ioObjTrig,portAddress,trigger);
        % reset to zero after 50 ms
        WaitSecs(0.05);
        io64(ioObjTrig,portAddress,0);
    end
    if EYE
        % Send trigger to Eyelink
        Eyelink('Message', ['Trigger_' int2str(trigger)]);
    end
end
function el = PF_EyelinkSetup(cfg,var_el)
    %% SET-UP
    if var_el.setup
        % Do all the Eyelink set-up
        % http://psychtoolbox.org/docs/EyelinkDemos
        % https://github.com/MEGSupportCHBH/TheCHBH/blob/master/MEGEyelinkDemoCodeCHBH/Eyelink_Demo.m
        % http://sr-research.jp/support/manual/EyeLink%20Programmers%20Guide.pdf
        % http://sr-research.jp/support/EyeLink%201000%20User%20Manual%201.5.0.pdf
        %
        % Provide Eyelink with details about the graphics environment
        % and perform some initializations. The information is returned
        % in a structure that also contains useful defaults
        % and control codes (e.g. tracker state bit and Eyelink key values).
        el=EyelinkInitDefaults(cfg.window);
        % update the defaults
        el.calibrationtargetsize   = 1; % size of calibration target as percentage of screen (original 2.5)
        el.calibrationtargetwidth  = 0.5; % width of calibration target's border as percentage of screen (original 1)
        el.targetbeep              = 0; % no beep when a target is presented (original 1)
        el.feedbackbeep            = 0; % no beep after calibration/drift correction (original 1)
        el.displayCalResults       = 1; % (original 0)
        el.eyeimagesize            = 50;  % percentage of screen (original 30)
        EyelinkUpdateDefaults(el);
        % Initialization of the connection with the Eyelink Gazetracker.
        % exit program if this fails.
        if ~EyelinkInit
            fprintf('Eyelink Init aborted.\n');
            cleanup;  % cleanup function
            return;
        else
            disp('Eyelink initizalized')
        end
        % Set data in samples sent through link (GAZE = screen xy (gaze)
        % position, GAZERES = units-per-degree screen resolution, AREA = pupil
        % area, STATUS = warning and error flags, INPUT = input port data lines)
        Eyelink('Command', 'link_sample_data = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'); 
        % Sets the gaze-position coordinate system, which is used for all 
        % calibration target locations and drawingcommands. Usually set to 
        % correspond to the pixel mapping of the subject display
        Eyelink('Command','screen_pixel_coords = %ld %ld %ld %ld',  cfg.windowRect(1),cfg.windowRect(2),cfg.windowRect(3),cfg.windowRect(4));
        % standard messages for the EFD file (DISPLAY_COORDS = display
        % coordinate system, FRAMERATE = display refresh rate)
        Eyelink('message','DISPLAY_COORDS %ld %ld %ld %ld',         cfg.windowRect(1),cfg.windowRect(2),cfg.windowRect(3),cfg.windowRect(4));
        Eyelink('message','FRAMERATE %d Hz.',                    cfg.frameRate);
        % Use Psychophysical tracker configuration. The psychophysical
        % configuration is useful for neurological and smooth-pursuit research, and
        % reports very small saccades. It also better estimates saccade durations and
        % average velocities. 
        Eyelink('Command', 'recording_parse_type = GAZE');
        Eyelink('Command', 'saccade_velocity_threshold = 22');
        Eyelink('Command', 'saccade_acceleration_threshold = 3800');
        Eyelink('Command', 'saccade_motion_threshold = 0.0');
        Eyelink('Command', 'saccade_pursuit_fixup = 60');
        Eyelink('Command', 'fixation_update_interval = 0');
        % Eyelink uses an online heuristic filter to decrease noise in the data
        % output. However, this increases the tracker delay froma an eye
        % movement to the data (in the 1000Hz mode: filter off: < 2 ms, 
        % filter level 1 (standard: < 3 ms, filter level 2 (extra) < 4 ms). The
        % heuristic filtering is turned off here, check what is recommended for
        % your experiment.
        Eyelink('Command', 'heuristic_filter = 0');
        % if YES = pupil area is converted to diameter, if NO, output is pupil
        % area.
        Eyelink('Command', 'pupil_size_diameter = YES');
        if strcmp(var_el.eye,'LEFT')
            % just record from left eye
            % YES for binocular tracking, NO for monocular tracking
            Eyelink('Command', 'binocular_enabled = NO');
            % select the eye to record from
            Eyelink('Command', 'active_eye = LEFT');
            % Sets which types of events will be written to EDF file or sent 
            % through link 
            % (LEFT, RIGHT = events for one or both eyes, FIXATION = fixation 
            % start and end events, SACCADE = saccade start and end, BLINK = blink 
            % start an end, MESSAGE = messages (user notes in file), INPUT  = 
            % changes in input port lines)
            Eyelink('Command', 'file_event_filter = LEFT,FIXATION,SACCADE,BLINK,MESSAGE,INPUT');
            Eyelink('Command', 'link_event_filter = LEFT,FIXATION,FIXUPDATE,SACCADE,BLINK,MESSAGE,INPUT');
        elseif strcmp(var_el.eye,'RIGHT')
            % just records from left eye
            % YES for binocular tracking, NO for monocular tracking
            Eyelink('Command', 'binocular_enabled = NO');
            % select the eye to record from
            Eyelink('Command', 'active_eye = RIGHT');
            % Sets which types of events will be written to EDF file or sent 
            % through link 
            % (LEFT, RIGHT = events for one or both eyes, FIXATION = fixation 
            % start and end events, SACCADE = saccade start and end, BLINK = blink 
            % start an end, MESSAGE = messages (user notes in file), INPUT  = 
            % changes in input port lines)
            Eyelink('Command', 'file_event_filter = RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,INPUT');
            Eyelink('Command', 'link_event_filter = RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,MESSAGE,INPUT');
        else
            % record from both eyes
            % YES for binocular tracking, NO for monocular tracking
            Eyelink('Command', 'binocular_enabled = YES');
        end
        % Sets data in samples written to EDF file.(GAZE = screen xy (gaze)
        % position, GAZERES = units-per-degree screen resolution, HREF = 
        % head-referenced gaze, PUPIL = raw eye camera pupil coordinates, AREA 
        % = pupil area, STATUS = warning and error flags, INPUT = input port 
        % data lines)
        Eyelink('Command', 'file_sample_data  = GAZE,GAZERES,HREF,PUPIL,AREA,STATUS,INPUT');
        % Select the method used to fit the pupil and determine pupil position.
        % The Centroid mode tracks the center of the thresholded pupil
        % using a center of mass algorithm whereas the Ellipse mode determines the
        % center of the pupil by fitting an ellipse to the thresholded pupil mass.
        % For most purposes, the centroid algorithm is recommended as it has very low
        % noise. However, if the pupil may be significantly occluded (for example by the
        % eyelids) the ellipse fitting algorithm may give a more accurate estimation of
        % pupil position.
        Eyelink('Command', 'use_ellipse_fitter = NO');
        % set the sampling rate to 1000 Hz
        Eyelink('Command', 'sample_rate = 1000');
        % make sure we're still connected.
        if Eyelink('IsConnected')~=1 && input.dummymode == 0
            exit_flag = 'ESC';
            return;
        end
        % open file for recording data on the tracker computer
        status = Eyelink('Openfile', var_el.edfname);
        if ~status
            disp('EDF file opened on Eyelink computer')
        else
            error(['Could not open EDF file on Eyelink computer, error: ' int2str(status)])
        end
    end
    %% CALIBRATION
    if var_el.cali
        % Do setup and calibrate the eye tracker
        EyelinkDoTrackerSetup(el);
    end
    %% DRIFT CORRECTION
    if var_el.drift
        % do a final check of calibration using drift correction
        % You have to hit esc before return.
        % this try/catch loop is a lazy way of fixing that the first drift
        % correction requires "el" and the second one "var_el.el".
        try
            EyelinkDoDriftCorrection(el);
        catch
            EyelinkDoDriftCorrection(var_el.el);
        end
    end
    %% START RECORDING
    % Start recording eye position
    Eyelink('StartRecording');
    % record a few samples before we actually start displaying
    WaitSecs(0.5);
    Eyelink('message','EXP_START')
    WaitSecs(0.5);
end
function PF_EyelinkClose(var_el)
    % finish up: stop recording eye-movements, 
    % close graphics window, close data file and shut down tracker
    Eyelink('message','EXP_END');
    WaitSecs(0.5);
    Eyelink('StopRecording');
    Eyelink('CloseFile');   
    % transfer file from tracker computer to experiment directory
    Eyelink('ReceiveFile',var_el.edfname,var_el.edffolder,1); 
    % Shutdown Eyelink:
    Eyelink('Shutdown');
end
%% Make the input files for the participants
clear variables
% turn off this warning
warning('off','MATLAB:table:RowsAddedExistingVars')

%% Variables
var.cats        = {'animals','clothing','food','plants','vehicles'}; % the object categories
var.objLoc      = 'Pics'; % location of the images
var.objCatN     = 1500; % number of images per category available
var.conds       = {'prime0b','prime1b','prime2b'}; % no prime, prime 1-back, prime 2-back
var.condsN      = 480; % number of trials per prime condition
var.ImgPara     = 1:4; % number of parafoveally presented image per trial
var.ImgFov      = 1; % number of foveally presented image per trial
var.subjN       = 101:160;
var.totN        = var.condsN*length(var.conds);
var.blocks      = (var.totN/15) / 8; % it takes 15 trials to have one in every combination of conditions and categories, so maximally 100 blocks, but you can make this less by deviding it by X, in this case 8.
var.blockN      = var.totN/var.blocks; % amount of trials per block
var.m_rep       = 2; % number of times an item is picked from the same category and condition (see next two lines)
var.m_fovN      = length(var.cats)*length(var.conds)*var.m_rep; % memory task, number of foveal items per block
var.m_newN      = length(var.cats)*length(var.conds)*var.m_rep; % memory task, number of novel items per block
var.m_totN      = (var.m_fovN+var.m_newN)*var.blocks; % total amount of memory trials
% triggers
% parafov onset: first screen:  no prime (1), prime 1-back (129), prime 2-back (65)
%                second screen: no prime (2), prime 1-back (130), prime 2-back (66) 
var.trg.pfov    = [1 129 65;... % 00000001, 10000001, 01000001
                   2 130 66];   % 00000010, 10000010, 01000010
% foveal onset: animal (192), clothing (160), food (144), plants (136), vehicles (132)
var.trg.fov     = [192 160 144 136 132]; % 11000000, 10100000, 10010000, 10001000, 10000100
% memory item onset: old (24), new (36)
var.trg.mem     = [24, 36]; % 00011000 & 00100100

%% Set-up
% get the images per category
for c=1:length(var.cats)
    var.imgs.(var.cats{c}) = dir(fullfile(var.objLoc,var.cats{c},'*.jpg'));
end
% get the objects per category
for c=1:length(fieldnames(var.imgs))
    var.objs.(var.cats{c}) = regexprep({var.imgs.(var.cats{c}).name},'[_]\d*[\w].jpg','');
end
clear c
% get the unique objects per category
for c=1:length(var.cats)
    var.objs_uni.(var.cats{c}) = unique(var.objs.(var.cats{c}));
end

%% Main
% loop over the participants
for s=var.subjN
    % if there is no imput file for the current participant, make it.
    files = dir('SubjectFiles');
    do_subj = ~any(strcmp({files.name},[num2str(s) '_Ret.mat']));
    clear files
    if do_subj
        fprintf(['\nCurrent participant: ' num2str(s)]);
        %% start of getting the main task list
        cur_objs_uni = var.objs_uni; % will be used to shuffle the object names
        % save all the variables so that items can be chosen from it for this
        % participant
        cur_imgs = var.imgs;
        % get the images presented at the fovea for the three different
        % conditions. 100 per category (100*5=500), per condition (500*3=1500)
        for c=1:length(var.cats)
            for c2=1:length(var.conds)
                % shuffle the object names
                cur_objs_uni.(var.cats{c}) = cur_objs_uni.(var.cats{c})(randperm(length(cur_objs_uni.(var.cats{c}))));
                for i=1:length(var.objs_uni.(var.cats{c}))
                    % find all the imagees associated with the current object
                    tmp = find(contains({cur_imgs.(var.cats{c}).name},cur_objs_uni.(var.cats{c}){i}));
                    if length(tmp) > 15
                        error(['duplicate object name: ',cur_imgs.(var.cats{c})(tmp(1)).name,' & ',cur_imgs.(var.cats{c})(tmp(end)).name])
                    end
                    % select a random object image
                    cur_fov.(var.conds{c2}).(var.cats{c}){i} = cur_imgs.(var.cats{c})(tmp(randi(length(tmp),1))).name;
                    % remove that image from all images
                    cur_imgs.(var.cats{c})(contains({cur_imgs.(var.cats{c}).name},(cur_fov.(var.conds{c2}).(var.cats{c}){i})),:)=[];
                    clear tmp
                end
                clear i
            end
            clear c2
        end
        clear c
        % get an array of all the conditions in a way that makes sure there are no
        % more than 2 consecutive instances of the same condition
        conds_all=[];
        for c=1:var.condsN
            conds2      = [var.conds];
            conds_all   = [conds_all conds2(randperm(length(conds2)))];
            clear conds2
        end
        clear c
        % create the order of foveal images, so that there are an equal amount
        % per category and per block.
        cur_blCatN = var.blockN/length(var.cats)/length(var.conds); % get the amount of images from the same category, per condition
        fovs_all   = cell(var.totN,1);
        % loop over the experimental blocks
        for b=1:var.blocks
            do_block = true;
            while do_block
                % get the foveal images for this block per condition
                for c2=1:length(var.conds)
                    cur_blImg.(var.conds{c2})  = [];
                end
                clear c2
                for c=1:length(var.cats)
                    for c2=1:length(var.conds)
                        cur_blImg.(var.conds{c2}) = [cur_blImg.(var.conds{c2}) [cur_fov.(var.conds{c2}).(var.cats{c})(cur_blCatN*(b-1)+1:cur_blCatN*b); repmat(var.cats(c),1,cur_blCatN)]];
                    end
                    clear c2
                end
                clear c
                % shuffle this blocks foveal images
                for c2=1:length(var.conds)
                    cur_blImg.(var.conds{c2}) = cur_blImg.(var.conds{c2})(:,randperm(size(cur_blImg.(var.conds{c2}),2)));
                end
                clear c2
                % place the chosen images in the right order and check if there are no
                % more than 4 consecutive trials with the same object category
                do_block = false; % complete this block if this is the case
                for t=var.blockN*(b-1)+1:var.blockN*b
                    fovs_all(t,1:2) = cur_blImg.(conds_all{t})(:,1)';
                    if t>3
                        if isequal(fovs_all(t,2),fovs_all(t-1,2),fovs_all(t-1,2),fovs_all(t-3,2))
                            do_block = false; % there are more than 4 consecutive trials with the same object category, so try this block again
                        end
                    end
                    cur_blImg.(conds_all{t})(:,1) =[];
                end
                clear t cur_blImg
            end
        end
        clear b do_block cur_blCatN cur_fov cur_objs_uni
        % make placeholders for the parafoveal stimuli per condition
        % *** prime0b:
        % get all the possible combinations
        tmp = perms(var.cats);
        % get all the repetitions that fully fit in the trials
        tmpn1 = length(tmp)*floor(var.condsN/length(tmp));
        % and the remainder
        % var.condsN-tmpn1;
        % select the best option from the remainder and not
        % randomized so it is the same across participants.
        % !!!
        % !!! done manually !!! so if you change this script later adjust this next line
        % !!!
        tmp2 = [];
        % repeat them for the total number of trials per condition
        tmp = [tmp(mod(0:tmpn1-1,length(tmp))+1,:); tmp2];
        % shuffle these
        tmp = tmp(randperm(length(tmp)),:);
        % loop over the foveal images and select a corresponding row from tmp
        tmpf = fovs_all(strcmp(conds_all,'prime0b'),2);
        for i=1:length(tmpf)
            idx = find(ismember(tmp(:,5),tmpf{i}),1,'first');
            tmp3(i,:) = tmp(idx,:);
            tmp(idx,:) = [];
        end
        pfovs.prime0b = tmp3(:,var.ImgPara);
        clear tmp* i idx
        % *** prime1b:
        % get all the possible combinations
        tmp.t = perms(var.cats);
        % select as many columns as the number of parafoveal images presented
        tmp.t = tmp.t(:,var.ImgPara);
        tmp.t = table2cell(unique(cell2table(tmp.t)));
        % get all the repetitions that fully fit in the trials
        tmpn1 = length(tmp.t)*floor(var.condsN/length(tmp.t));
        % and the remainder
        % var.condsN-tmpn1;
        % select the best options from the remainder and not
        % randomized so it is the same across participants.
        % !!!
        % !!! done manually !!! so if you change this script later adjust the lines containing tmpr
        % !!!
        tmpr = [];
        % repeat them for half the number of trials per condition
        % we take half so we can use one for the images presented on the
        % left (3) and one for the images presented on the right (4).
        tmp.t0 = [tmp.t(mod(0:(tmpn1/2)-1,length(tmp.t))+1,:); tmpr];
        % shuffle these
        tmp.t0 = tmp.t0(randperm(length(tmp.t0)),:);
        % and repeat
        tmp.t1 = [tmp.t(mod(0:(tmpn1/2)-1,length(tmp.t))+1,:); tmpr];       
        tmp.t1 = tmp.t1(randperm(length(tmp.t1)),:);
        % loop over the foveal images and select a corresponding row from
        % tmp.tx alternating between the left (3) and right (4) items
        tmpf = fovs_all(strcmp(conds_all,'prime1b'),2);
        % sort the foveal images to loop over them
        [~,I] = sort(tmpf);
        cnt = 1;
        for i=I'
            idx = find(any(ismember(tmp.(['t' num2str(mod(cnt,2))])(:,mod(cnt,2)+3),tmpf{i}),2),1,'first');
            tmp3(i,:) = tmp.(['t' num2str(mod(cnt,2))])(idx,:);
            tmp.(['t' num2str(mod(cnt,2))])(idx,:) = [];
            cnt = cnt+1;
        end
        pfovs.prime1b = tmp3(:,var.ImgPara);
        clear tmp* i idx I
        % *** prime2b:
        % get all the possible combinations
        tmp.t = perms(var.cats);
        % select as many columns as the number of parafoveal images presented
        tmp.t = tmp.t(:,var.ImgPara);
        tmp.t = table2cell(unique(cell2table(tmp.t)));
        % get all the repetitions that fully fit in the trials
        tmpn1 = length(tmp.t)*floor(var.condsN/length(tmp.t));
        % and the remainder
        % var.condsN-tmpn1;
        % select the best options from the remainder and not
        % randomized so it is the same across participants.
        % !!!
        % !!! done manually !!! so if you change this script later adjust the lines containing tmpr
        % !!!
        tmpr = [];
        % repeat them for half the number of trials per condition
        % we take half so we can use one for the images presented on the
        % left (1) and one for the images presented on the right (2).
        tmp.t0 = [tmp.t(mod(0:(tmpn1/2)-1,length(tmp.t))+1,:); tmpr];
        % shuffle these
        tmp.t0 = tmp.t0(randperm(length(tmp.t0)),:);
        % and repeat
        tmp.t1 = [tmp.t(mod(0:(tmpn1/2)-1,length(tmp.t))+1,:); tmpr];       
        tmp.t1 = tmp.t1(randperm(length(tmp.t1)),:);
        % loop over the foveal images and select a corresponding row from
        % tmp.tx alternating between the left (3) and right (4) items
        tmpf = fovs_all(strcmp(conds_all,'prime2b'),2);
        % sort the foveal images to loop over them
        [~,I] = sort(tmpf);
        cnt = 1;
        for i=I'
            idx = find(any(ismember(tmp.(['t' num2str(mod(cnt,2))])(:,mod(cnt,2)+1),tmpf{i}),2),1,'first');
            tmp3(i,:) = tmp.(['t' num2str(mod(cnt,2))])(idx,:);
            tmp.(['t' num2str(mod(cnt,2))])(idx,:) = [];
            cnt = cnt+1;
        end
        pfovs.prime2b = tmp3(:,var.ImgPara);
        clear tmp* i idx I cnt
        % loop over the trials and select the parafoveal images
        cnt.prime0b = 1;
        cnt.prime1b = 1;
        cnt.prime2b = 1;
        for t=1:var.totN
            % get the categories of the parafoveal images
            cur_pfov_cats = pfovs.(conds_all{t})(cnt.(conds_all{t}),:);
            if strcmp(conds_all{t},'prime0b') % fov image not primed by same object image
                % select one object image at random per category
                for c=1:length(cur_pfov_cats)
                    cur_pfov_imgs{c} = cur_imgs.(cur_pfov_cats{c})(randi(length(cur_imgs.(cur_pfov_cats{c})),1)).name;
                    % and remove that image from the pool
                    cur_imgs.(cur_pfov_cats{c})(contains({cur_imgs.(cur_pfov_cats{c}).name},cur_pfov_imgs{c}),:)=[];
                end
                % store the selected images in the right position
                pfovs_all(t,var.ImgPara) = cur_pfov_imgs;
                % also store the associated categories
                pfovs_all(t,length(var.ImgPara)+1:2*length(var.ImgPara)) = cur_pfov_cats;
            elseif strcmp(conds_all{t},'prime1b') || strcmp(conds_all{t},'prime2b') % fov image primed by same object image one screen back
                % select the foveal image and fill the rest with a random
                % category image
                cur_pfov_imgs = cell(1,length(var.ImgPara));
                cur_pfov_imgs(strcmp(cur_pfov_cats,fovs_all(t,2))) = fovs_all(t,1);
                for c=find(cellfun(@isempty,cur_pfov_imgs))
                    cur_pfov_imgs{c} = cur_imgs.(cur_pfov_cats{c})(randi(length(cur_imgs.(cur_pfov_cats{c})),1)).name;
                    % and remove that image from the pool
                    cur_imgs.(cur_pfov_cats{c})(contains({cur_imgs.(cur_pfov_cats{c}).name},cur_pfov_imgs{c}),:)=[];
                end
                % store the selected images in the right position
                pfovs_all(t,var.ImgPara) = cur_pfov_imgs;
                % also store the associated categories
                pfovs_all(t,length(var.ImgPara)+1:2*length(var.ImgPara)) = cur_pfov_cats;
            end
            cnt.(conds_all{t}) = cnt.(conds_all{t}) + 1;
            clear cur_pfov_* cur_img c idx
        end
        clear cnt* pfovs t
        % create an empty table
        T           = table;
        T.ppn       = repmat(s,[var.totN,1]);                       % participant number
        T.trialN    = [1:var.totN]';                                % trial number
        T.blockN    = sort(repmat(1:var.blocks,[1,var.blockN]))';   % block number
        T.cond      = conds_all';                                   % experimental condition
        T.fov_img   = fovs_all(:,1);                                % foveal image
        T.pfov1_img = pfovs_all(:,1);                               % parafoveal image first left
        T.pfov2_img = pfovs_all(:,2);                               % parafoveal image first right
        T.pfov3_img = pfovs_all(:,3);                               % parafoveal image second left
        T.pfov4_img = pfovs_all(:,4);                               % parafoveal image second right
        T.fov_cat   = fovs_all(:,2);                                % foveal category
        T.pfov1_cat = pfovs_all(:,5);                               % parafoveal category first left
        T.pfov2_cat = pfovs_all(:,6);                               % parafoveal category first right
        T.pfov3_cat = pfovs_all(:,7);                               % parafoveal category second left
        T.pfov4_cat = pfovs_all(:,8);                               % parafoveal category second right
        % add the triggers
        for t=1:size(T,1)
            if strcmp(T.cond{t},'prime0b')
                T.trg_pfov1(t) = var.trg.pfov(1,1);    % no prime
                T.trg_pfov2(t) = var.trg.pfov(2,1);    % no prime
            elseif strcmp(T.cond{t},'prime1b')
                T.trg_pfov1(t) = var.trg.pfov(1,2);    % prime 1-back
                T.trg_pfov2(t) = var.trg.pfov(2,2);    % prime 1-back
            elseif strcmp(T.cond{t},'prime2b')
                T.trg_pfov1(t) = var.trg.pfov(1,3);    % prime 2-back
                T.trg_pfov2(t) = var.trg.pfov(2,3);    % prime 2-back
            end
            if strcmp(T.fov_cat{t},'animals')
                T.trg_fov(t)   = var.trg.fov(1);
            elseif strcmp(T.fov_cat{t},'clothing')
                T.trg_fov(t)   = var.trg.fov(2);
            elseif strcmp(T.fov_cat{t},'food')
                T.trg_fov(t)   = var.trg.fov(3);
            elseif strcmp(T.fov_cat{t},'plants')
                T.trg_fov(t)   = var.trg.fov(4);
            elseif strcmp(T.fov_cat{t},'vehicles')
                T.trg_fov(t)   = var.trg.fov(5);
            end
        end
        save(['SubjectFiles' filesep num2str(s) '_ParaFov'],'T')
        clear *_all t
        %% start working on the memory list
        % make an empty memory table
        M = table;
        % loop over the blocks
        % when no item can be selected (due to randomness), try again
        do_loop = true;
        err_cnt = 0;
        while do_loop
            try
                for b=1:var.blocks
                    % get the current block trials
                    cur_t = T(1+((b-1)*var.blockN):var.blockN*b,:);
                    % make an empty table
                    tmp = table;
                    % make the trial counter for all trials
                    m_trl = 1;
                    % loop over the conditions
                    for co=1:length(var.conds)
                        % get the current condition trials
                        cur_tc = cur_t(strcmp(cur_t.cond,var.conds{co}),:);
                        % loop over the categories
                        for ca=1:length(var.cats)
                            % get the foveal images
                            cur_tcc_fov = cur_tc(strcmp(cur_tc.fov_cat,var.cats{ca}),:);
                            % choose the number of items based on the repetition value
                            idx = randperm(size(cur_tcc_fov,1));
                            cur_tcc_fov = cur_tcc_fov(idx(1:var.m_rep),:);
                            % save the chosen items in a table
                            for r=1:var.m_rep
                                tmp.ppn(m_trl) = cur_tcc_fov.ppn(r);
                                tmp.blockN(m_trl) = cur_tcc_fov.blockN(r);
                                tmp.cond(m_trl) = cur_tcc_fov.cond(r);
                                tmp.cat(m_trl) = cur_tcc_fov.fov_cat(r);
                                tmp.type(m_trl) = {'fov'};
                                tmp.item(m_trl) = cur_tcc_fov.fov_img(r);
                                m_trl = m_trl +1;
                            end
                            % remove those from the pool
                            cur_tc(ismember(cur_tc.trialN,cur_tcc_fov.trialN),:) =[];
                            clear cur_tcc* idx cur_c
                        end
                        clear cur_tc
                    end
                    % get the novel images
                    for ca=1:length(var.cats)
                        % get the images that have not been used yet from the current category
                        cur_c = {cur_imgs.(var.cats{ca}).name};
                        % remove the objects that are already old items
                        cur_o = extractBefore(tmp.item(strcmp(tmp.cat,var.cats{ca})),'_');
                        cur_c(contains(cur_c,cur_o))=[];
                        % at random, select only one image per object
                        for i=1:length(var.objs_uni.(var.cats{ca}))
                            idxs = find(contains(cur_c,var.objs_uni.(var.cats{ca}){i}));
                            idx = randperm(length(idxs));
                            cur_c(idxs(idx(2:end))) = [];
                            clear idx*
                        end
                        % get all the (random) images needed for this category in this block (total number it new items this block/number of categories)
                        idx = randperm(length(cur_c));
                        cur_c = cur_c(idx(1:var.m_newN/length(var.cats)));
                        tmpn = table;
                        tmpn.ppn(1:var.m_newN/length(var.cats))         = s;
                        tmpn.blockN(1:var.m_newN/length(var.cats))      = b;
                        tmpn.cond(1:var.m_newN/length(var.cats))        = {'new'};
                        tmpn.cat(1:var.m_newN/length(var.cats))         = var.cats(ca);
                        tmpn.type(1:var.m_newN/length(var.cats))        = {'new'};
                        tmpn.item(1:var.m_newN/length(var.cats))        = cur_c;
                        tmp                                             = [tmp; tmpn];
                        clear cur_c idx tmpn
                    end
                    % shuffle the items of this block
                    tmp = tmp(randperm(size(tmp,1)),:);
                    % add the trial numbers
                    tmp.trialN = [1+(b-1)*size(tmp,1):size(tmp,1)*b]';
                    tmp = tmp(:,[1 7 2:6]);
                    M = [M;tmp];
                    clear cur_t tmp
                    % exit the loop
                end
                do_loop = false;
            catch
                clear cur_t* b ca co idx r tmp m_trl
                M = table;
                m_trl_p = 1;
                err_cnt = err_cnt+1;
                if err_cnt == 10
                    error('could not complete loop for this participant, try to select the stimuli again for this participant')
                end
            end       
        end
        % add the triggers
        M.trg_mem(~strcmp(M.cond,'new')) = var.trg.mem(1); % old
        M.trg_mem(strcmp(M.cond,'new')) = var.trg.mem(2); % new
        % save the memory list
        save(['SubjectFiles' filesep num2str(s) '_Ret'],'M')
        clear m_* r b ca co do_loop cur_*
    end
    clear T M err_cnt do_subj
end
clear s var
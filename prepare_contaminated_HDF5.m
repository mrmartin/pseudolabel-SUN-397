if(~exist('train_set_proportion'))
    train_set_proportion=0.7;
end

%you need lots of swap to hold this, make sure the SSD is connected!
clearvars -except train_set_proportion

train_set_proportion

results_folder = 'pseudolabels_full_SUN_google_var_min_belief'
mkdir(results_folder)
%% load data %%
disp('loading labels')
home = pwd;
load labels
% cd '../../Datasets/SUN397/SUN_6_scenes/227x227/4096'
% %load data
% preds=dir('*_pred');
% image_data=struct([]);
% 
% all_features=[];
% all_labels=[];
% for i=1:length(preds)
%    tic;disp(['loading ' preds(i).name]);
%    thisone=csvread(preds(i).name);
%    image_data(i).features=thisone;
%    image_data(i).labels=i*ones(size(thisone,1),1);
% 
%    all_features=[all_features;thisone];
%    all_labels=[all_labels;i*ones(size(thisone,1),1)];
%    toc
%    %pause(1)%gives matlab time to display size of current variables in memory
% end
% 
% cd(home);
% clear thisone
%tic;load /media/martin/ssd-ext4/features_and_labels;toc
classes=dir('SUN_397_fc7/*.mat');
classes=struct2cell(classes);    
classes(2:end,:)=[];
%% define my classes
for i=1:length(classes)
    fprintf('SUN_label: %d - %s\n',histc(sun_label,i),classes{i}(1:end-4));
    fprintf('google_label: %d - %s\n',histc(goo_label,i),classes{i}(1:end-4));
end
disp(['number of SUN images per category is at least ' num2str(min(histc(sun_label,unique(sun_label))))])
disp(['number of google images per category is at least ' num2str(min(histc(goo_label,unique(goo_label))))])

%load SUN dataset features
sun_all_feat=single([]);
for i=1:length(classes)
	tmp=load(['SUN_397_fc7/' classes{i}]);
	sun_all_feat=[sun_all_feat;single(tmp.sun_feat)];
end
clear tmp
disp('loaded SUN dataset features');
whos sun_all_feat

for i=1:length(classes)
	tmp=load(['/media/martin/MartinK3TB/Experiments/pseudolabel_SUN_397/google_397_fc7/' classes{i}]);
	goo_feat{i}=single(tmp.goo_feat);
end
clear tmp
whos goo_feat

%%

for min_belief=[5 2 1] %12:-1:1
for diss_set_size=[50 0 1 5]%5:25:100 %this is the proportion in each class! 10:50:450
    for train_set_size=50%this is now set size! 5:25:80 %this is the porportion in each class! 10:50:450
		if((diss_set_size==0 && min_belief==10) || diss_set_size~=0)
			image_orders=struct([]);

			disp(['sun_all_feats and sun_label are ordered. Let''s randomly select ' num2str(train_set_size) ' images from each SUN class for training and ' num2str(diss_set_size) '% of each google class for dissolution, and save indexes in a variable'])
			sun_train_order=[];
			sun_test_order=[];
			for i=1:length(classes)
				rng default
				set_order = randperm(histc(sun_label,i));
				image_orders(i).train_set = set_order(1:train_set_size);%set_order(1:floor(train_set_size/100*length(set_order)));
				image_orders(i).test_set = set_order((train_set_size+1):end);%set_order(ceil(train_set_size/100*length(set_order)):end);
				%careful, this code assumes that sun_label is ordered (implies a reorganization of sun_all_feat, which here has been implicitely done)
				sun_train_order=[sun_train_order image_orders(i).train_set+max([0;find(sun_label==i-1)])];
				sun_test_order=[sun_test_order image_orders(i).test_set+max([0;find(sun_label==i-1)])];
				image_orders(i).diss_set=1:floor(diss_set_size/100*histc(goo_label,i));%the diss set order is not random, because the best images are first
				if(length(image_orders(i).diss_set)==0)
					image_orders(i).diss_set=1;
				end
			end
			%image_orders has the orders in each class. sun_train_order and sun_test_order have orders in sun_all_feat and sun_label
			image_orders

			%% save data for itteration 0, where no contaminated data are used in training
			delete /media/martin/ssd-ext4/sun_dataset.h5

			%prepare h5 dataset file, and fill it with data
			h5create('/media/martin/ssd-ext4/sun_dataset.h5','/data',[1 1 4096 length(sun_train_order)],'DataType','single')
			h5create('/media/martin/ssd-ext4/sun_dataset.h5','/label',[1 length(sun_train_order)],'DataType','single')

			h5write('/media/martin/ssd-ext4/sun_dataset.h5','/data',reshape(sun_all_feat(sun_train_order,:)',[1,1,4096,length(sun_train_order)]))
			h5write('/media/martin/ssd-ext4/sun_dataset.h5','/label',single(sun_label(sun_train_order)-1)')%INDEX FROM 0

			%test_order = randperm(length(test_labels));
			delete /media/martin/ssd-ext4/sun_test_dataset.h5

			%prepare h5 dataset file, and fill it with data
			h5create('/media/martin/ssd-ext4/sun_test_dataset.h5','/data',[1 1 4096 length(sun_test_order)],'DataType','single')
			h5create('/media/martin/ssd-ext4/sun_test_dataset.h5','/label',[1 length(sun_test_order)],'DataType','single')

			h5write('/media/martin/ssd-ext4/sun_test_dataset.h5','/data',reshape(sun_all_feat(sun_test_order,:)',[1,1,4096,length(sun_test_order)]))
			h5write('/media/martin/ssd-ext4/sun_test_dataset.h5','/label',single(sun_label(sun_test_order)-1)')%INDEX FROM 0

			%diss_order = randperm(length(diss_labels));
			delete /media/martin/ssd-ext4/sun_contaminated_dataset.h5
			%because the google features don't fit in memory, create the contaminated.h5 files by appending
			h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',[1 1 4096 size([image_orders(:).diss_set],2)],'DataType','single')
			h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',[1 size([image_orders(:).diss_set],2)],'DataType','single')

			first_next_elem=1;
			for i=1:length(classes)
				%loop through, load each from .mat, then append the appropriate ones to h5 dataset
				%tmp=load(['/media/martin/MartinK3TB/Experiments/pseudolabel_SUN_397/google_397_fc7/' classes{i}]);
				%append the subset defined by image_orders(:).diss_set
				h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',reshape(single(goo_feat{i}(image_orders(i).diss_set,:)'),[1 1 4096 length(image_orders(i).diss_set)]),[1 1 1 first_next_elem],[1 1 4096 length(image_orders(i).diss_set)]);
				h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',single((i-1)*ones(1,length(image_orders(i).diss_set))),[1 first_next_elem],[1 length(image_orders(i).diss_set)]);
				disp(['length of class ' num2str(i) ' chosen subset is ' num2str(length(image_orders(i).diss_set)) ' (starting at ' num2str(first_next_elem) ')'])
				first_next_elem=first_next_elem+length(image_orders(i).diss_set);
			end
			clear tmp;

			%create correspondences between each row in sun_contaminated_dataset.h5 and the image_order(i).diss_set(j)
			diss_correspondences=[];	
			for i=1:length(classes)
				diss_correspondences=[diss_correspondences;i*ones(length(image_orders(i).diss_set),1) image_orders(i).diss_set'];
			end

			%% run one itteration of caffe
			[~,m]=unix('rm SUN_solver_oneit.prototxt.log');
			it_size=1000;
			tmp=randperm(size([image_orders(:).diss_set],2));
			diss_testing=diss_correspondences(tmp(1:floor(length(tmp)/2)),:);%select random half for first itteration
			diss_training=diss_correspondences(tmp(floor(1+length(tmp)/2):end),:);
			clear diss_accuracy tmp diss_number
			tic
			for itteration=0:it_size:39000
				disp(['running itteration ' num2str(itteration)])

				[~,m]=unix('cp SUN_solver_oneit_train.prototxt tmp_matlab_solver');
				[~,m]=unix('echo "snapshot_prefix: \"SUN_matlab_solver\"" >> tmp_matlab_solver');
				[~,m]=unix(['echo "max_iter: ' num2str(itteration+it_size) '" >> tmp_matlab_solver']);
				[~,m]=unix('echo /media/martin/ssd-ext4/sun_test_dataset.h5 > matlab_dataset_test_list');
				successtrain=1;
				while(successtrain~=0)
				    if(itteration==0)
				        [successtrain,mtrain]=unix('GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/caffe train -solver=tmp_matlab_solver 2>> SUN_solver_oneit.prototxt.log');
				    else
				        [successtrain,mtrain]=unix(['GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/caffe train -solver=tmp_matlab_solver -snapshot=SUN_matlab_solver_iter_' num2str(itteration) '.solverstate 2>> SUN_solver_oneit.prototxt.log']);
				    end

				    if(successtrain~=0)
				        disp('error training, trying again!');
				        successtrain
				    end
				end
				%%
				disp('rewriting diss set')
				delete /media/martin/ssd-ext4/sun_contaminated_dataset.h5
				%because the google features don't fit in memory, create the contaminated.h5 files by appending
				h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',[1 1 4096 length(diss_testing)],'DataType','single')
				h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',[1 length(diss_testing)],'DataType','single')

	%             first_next_elem=1;
	%             for i=1:length(classes)
	%                 chosen_labels=0*ones(1,sum(diss_testing(:,1)==i));%*(i-1) - true labels are not given to testing
	%                 if(length(chosen_labels)>0)
	% 		            %loop through, load each from .mat, then append the appropriate ones to h5 dataset
	% 		            %tmp=load(['/media/martin/MartinK3TB/Experiments/pseudolabel_SUN_397/google_397_fc7/' classes{i}]);
	% 		            chosen_features=goo_feat{i}(diss_testing(diss_testing(:,1)==i,2),:);
	%                 
	%                     %append the subset defined by image_orders(:).diss_set
	%                     h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',reshape(single(chosen_features),[1 1 4096 length(chosen_labels)]),[1 1 1 first_next_elem],[1 1 4096 length(chosen_labels)]);
	%                     h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',single(chosen_labels),[1 first_next_elem],[1 length(chosen_labels)]);
	%                     %disp(['length of class ' num2str(i) ' chosen subset is ' num2str(length(chosen_labels)) ' (starting at ' num2str(first_next_elem) ')'])
	%                     first_next_elem=first_next_elem+length(chosen_labels);
	%                 end
	%             end
	%             clear tmp;

				chosen_labels=zeros(1,length(diss_testing));
				chosen_features=zeros(4096,length(diss_testing));
				
				for i=1:length(classes)
				    if(sum(diss_testing(:,1)==i)>0)
				        %chosen_labels(sum(diss_testing(:,1)<i)+1:sum(diss_testing(:,1)<=i))=0*ones(1,sum(diss_testing(:,1)==i));%*(i-1) - true labels are not given to testing
				        %loop through, load each from .mat, then append the appropriate ones to h5 dataset
				        chosen_features(:,sum(diss_testing(:,1)<i)+1:sum(diss_testing(:,1)<=i))=goo_feat{i}(diss_testing(diss_testing(:,1)==i,2),:)';
				    end
				end
				    
				%append the subset defined by image_orders(:).diss_set
				h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',reshape(single(chosen_features),[1 1 4096 length(chosen_labels)]),[1 1 1 1],[1 1 4096 length(chosen_labels)]);
				h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',single(chosen_labels),[1 1],[1 length(chosen_labels)]);

				clear chosen_features;
				%%
				chosen_indexes=[];%true indexes and labels for current diss test set
				chosen_labels=[];
				first_next_elem=1;
				for i=1:length(classes)
				    chosen_indexes=[chosen_indexes;first_next_elem+diss_testing(diss_testing(:,1)==i,2)];
				    chosen_labels=[chosen_labels;i*ones(sum(diss_testing(:,1)==i),1)];
				    first_next_elem=first_next_elem+sum(diss_testing(:,1)==i)+sum(diss_training(:,1)==i);
				end

				%evaluate on diss_set
				disp('evaluating on contaminated test subset')
				[~,m]=unix('rm -r contaminated_perceptron_extracted');
				[~,m]=unix('echo /media/martin/ssd-ext4/sun_contaminated_dataset.h5 > matlab_dataset_test_list');
				[~,mget]=unix(['GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/extract_features SUN_matlab_solver_iter_' num2str(itteration+it_size) '.caffemodel SUN_matlab_perceptron_extract.prototxt perceptron contaminated_perceptron_extracted ' num2str(ceil(length(diss_testing)/10))]);
				%%
				%for each class take the most likely from the diss set and put them in the
				%train set. These will be changed in the next itteration
				
				%sometimes, extract_features sticks a $ into the .csv. Remove it
				%[~,m]=unix('sed -i ''s/\$//g'' contaminated_perceptron_extracted.csv');
				%I don't know why, but sometimes the csvread line crashes because there are bad symbols. So let's run the sed twice
				%[~,m]=unix('sed -i ''s/[^-,\.0-9]//g'' contaminated_perceptron_extracted.csv');

				try
					diss_pred=csvread('contaminated_perceptron_extracted.csv');
				catch
					disp('csvread contaminated_perceptron_extracted.csv failed! Running sed ''s/[^-,\.0-9]//g'', and trying once again.');
					[~,m]=unix('sed -i ''s/[^-,\.0-9]//g'' contaminated_perceptron_extracted.csv');
					diss_pred=csvread('contaminated_perceptron_extracted.csv');
				end
				diss_pred=diss_pred(1:length(diss_testing),:);
				%how correct is it? (the learning process doesn't know this, but it's
				%interesting for analysis)
			   
				%this doesn't allow that confusion, and keeps a (the max value)

				actual_labels_for_current_diss_testing=chosen_labels;%diss_correspondences(diss_testing,1)';
				%where google set is actually i, find images where i>min_belief. Those will be used

				%the net may predict class a to be class b, which is irrelevant for us. Instead, we wish to find if it's class a or not
				[a,b]=max(diss_pred,[],2);
				diss_accuracy((itteration/it_size)+1) = sum(b(a>min_belief)==actual_labels_for_current_diss_testing(a>min_belief))/length(b(a>min_belief));
				
				%%
				new_diss_training_labels=[];%predicted labels
				new_training_indexes=[];%indexes of diss_correspondeces, where a(actual_labels==i)>min_belief
				
				for i=1:length(classes)
				    [a,b]=max(diss_pred(chosen_labels==i,:),[],2);
				    this_class_indexes=chosen_indexes(chosen_labels==i);
				    new_diss_training_labels=[new_diss_training_labels;b(a>min_belief)];%predicted labels
				    new_training_indexes=[new_training_indexes;this_class_indexes(a>min_belief)];
				end
				%I don't know how it happened, but there was a new_training_index value higher than length(diss_correspondences). Fix it
				new_diss_training_labels(new_training_indexes>length(diss_correspondences))=[];
				new_training_indexes(new_training_indexes>length(diss_correspondences))=[];
				
				%randomly remove 50% of chosen training images
				tmp=randperm(length(new_training_indexes));
				new_training_indexes=new_training_indexes(tmp(1:round(length(tmp)/2)));

				new_training_indexes_sorted=sort(new_training_indexes);
				new_training_indexes_sorted(1:10)'

				new_diss_training_labels=new_diss_training_labels(tmp(1:round(length(tmp)/2)));%predicted labels
				
				new_diss_training=diss_correspondences(new_training_indexes,:);

				new_testing_indexes=setdiff(1:length(diss_correspondences),new_training_indexes);%must contain true labels
				new_diss_testing=diss_correspondences(new_testing_indexes,:);

				%%
				diss_training=new_diss_training;%diss_testing(rand_diss_set(1:floor(length(b)/2)));%use half of what's been tested
				diss_testing=new_diss_testing;%[prev_diss_training diss_testing(rand_diss_set((floor(length(b)/2)+1):end))];%other half of what's been tested + previous training set

				[unique(diss_training) histc(diss_training, unique(diss_training))]'
				
				diss_number((itteration/it_size)+1) = length(diss_training);
				disp(['writing new sun_dataset.h5, including ' num2str(length(diss_training)) ' pseudolabel images'])
				%write new training file by writing training data, and appending selected diss data
				delete /media/martin/ssd-ext4/sun_dataset.h5
				%because the google features don't fit in memory, create the contaminated.h5 files by appending
				h5create('/media/martin/ssd-ext4/sun_dataset.h5','/data',[1 1 4096 length(diss_training)+length(sun_train_order)],'DataType','single')
				h5create('/media/martin/ssd-ext4/sun_dataset.h5','/label',[1 length(diss_training)+length(sun_train_order)],'DataType','single')
				%% training data:
				%h5write('/media/martin/ssd-ext4/sun_dataset.h5','/data',reshape(sun_all_feat(sun_train_order,:)',[1,1,4096,length(sun_train_order)]),[1 1 1 1],[1 1 4096 length(sun_train_order)]);
				%h5write('/media/martin/ssd-ext4/sun_dataset.h5','/label',single(sun_label(sun_train_order)-1)',[1 1],[1 length(sun_train_order)]);

				%% diss data:
	%             first_next_elem=1+length(sun_train_order);
	%             for i=1:length(classes)
	% 		        chosen_labels=new_diss_training_labels(diss_training(:,1)==i);%use labels predicted in previous itteration
	%                 if(length(chosen_labels)>0)
	% 		            %loop through, load each from .mat, then append the appropriate ones to h5 dataset
	% 		            %tmp=load(['/media/martin/MartinK3TB/Experiments/pseudolabel_SUN_397/google_397_fc7/' classes{i}]);
	% 		            chosen_features=goo_feat{i}(diss_training(diss_training(:,1)==i,2),:)';%find images according to their true labels
	% 		            %0*ones(1,sum(diss_testing(:,1)==i));%*(i-1) - true labels are not given to testing
	%                     %append the subset defined by image_orders(:).diss_set
	%                     h5write('/media/martin/ssd-ext4/sun_dataset.h5','/data',reshape(single(chosen_features),[1 1 4096 length(chosen_labels)]),[1 1 1 first_next_elem],[1 1 4096 length(chosen_labels)]);
	%                     h5write('/media/martin/ssd-ext4/sun_dataset.h5','/label',single(chosen_labels)',[1 first_next_elem],[1 length(chosen_labels)]);
	% 
	%                     %disp(['length of class ' num2str(i) ' chosen subset is ' num2str(length(chosen_labels)) ' (starting at ' num2str(first_next_elem) ')'])
	% 
	%                     first_next_elem=first_next_elem+length(chosen_labels);
	%                 end
	%             end

				chosen_labels=zeros(1,length(diss_training(:,1))+length(sun_train_order));
				chosen_labels(1:length(sun_train_order))=single(sun_label(sun_train_order)-1)';
				chosen_features=zeros(4096,length(diss_training(:,1))+length(sun_train_order));
				chosen_features(:,1:length(sun_train_order))=sun_all_feat(sun_train_order,:)';
				%whos('chosen_labels','chosen_features')
				
				for i=1:length(classes)
				    if(sum(diss_training(:,1)==i)>0)
				        chosen_labels(length(sun_train_order)+(sum(diss_training(:,1)<i)+1:sum(diss_training(:,1)<=i)))=new_diss_training_labels(diss_training(:,1)==i)';%*(i-1) - true labels are not given to testing
				        %loop through, load each from .mat, then append the appropriate ones to h5 dataset
				        chosen_features(:,length(sun_train_order)+(sum(diss_training(:,1)<i)+1:sum(diss_training(:,1)<=i)))=goo_feat{i}(diss_training(diss_training(:,1)==i,2),:)';
				    end
				end
				
				whos('chosen_labels','chosen_features')
				
				train_dataset_permutation=randperm(length(chosen_labels));
				    
				%append the subset defined by image_orders(:).diss_set
				h5write('/media/martin/ssd-ext4/sun_dataset.h5','/data',reshape(single(chosen_features(:,train_dataset_permutation)),[1 1 4096 length(chosen_labels)]),[1 1 1 1],[1 1 4096 length(chosen_labels)]);
				h5write('/media/martin/ssd-ext4/sun_dataset.h5','/label',single(chosen_labels(train_dataset_permutation)),[1 1],[1 length(chosen_labels)]);
				clear chosen_features;
				
				%%
				%[~,m]=unix('grep loss SUN_solver_oneit.prototxt.log | grep Iteration | sed -n ''n;p'' | sed ''s/^.*Iteration \([0-9]*\), loss = \([0-9\.]*\)$/\1, \2/g'' > train_loss.csv');
				[~,m]=unix('grep -B1 "accuracy = " SUN_solver_oneit.prototxt.log | grep -B1 Train | sed -e '':a'' -e ''N'' -e ''$!ba'' -e ''s/loss = [0-9\.]*\n//g'' | grep Iteration | sed ''s/.*Iteration \([0-9]*\).*accuracy = \([0-9\.]*\)/\1, \2/g'' > train_loss.csv');
				%[~,m]=unix('grep loss SUN_solver_oneit.prototxt.log | grep Iteration | sed -n ''p;n'' | sed ''s/^.*Iteration \([0-9]*\), loss = \([0-9\.]*\)$/\1, \2/g'' > test_loss.csv');
				[~,m]=unix('grep "accuracy = " SUN_solver_oneit.prototxt.log | grep Test | sed ''s/.*accuracy = //g'' > test_loss.csv');
				train_accuracy=csvread('train_loss.csv');
				test_accuracy=csvread('test_loss.csv');

				plot(train_accuracy(:,1),train_accuracy(:,2))
				hold on
				%plot(train_accuracy(:,1),test_accuracy(1:2:(2*size(train_accuracy,1))),'r')
				plot(train_accuracy(:,1),test_accuracy(1:end-(itteration/it_size)-1),'r')
				
				if(diss_set_size~=0)
					plot(train_accuracy(20:20:length(diss_accuracy)*20,1),diss_accuracy,'g')
					text(train_accuracy(20:20:length(diss_accuracy)*20,1),diss_accuracy,num2str(diss_number'),'horiz','center','vert','bottom')
				end

				legend('training accuracy','test accuracy',['contaminated accuracy where belief > ' num2str(min_belief)],'Location','northwest')
				ylim([0 1])
				hold off
				title([num2str(train_set_size) ' training images and ' num2str(diss_set_size) '% of the top images from google, with min belief ' num2str(min_belief)])
				xlabel('iterations')
				ylabel('accuracy')
				pause(3)
				toc
			end
			save([results_folder '/accuracy_' num2str(train_set_size) '_' num2str(diss_set_size) '_' num2str(min_belief)],'train_accuracy','test_accuracy','diss_number','diss_accuracy')
			print('-f1','-dpng',[results_folder '/accuracy_' num2str(train_set_size) '_' num2str(diss_set_size) '_' num2str(min_belief) '.png'])
		end
    end
end
end
%% resume from snapsot and run the next caffe itteration
% for itteration=0:100:10000
%     disp(['running itteration ' num2str(itteration)])
%     [~,m]=unix('cp SUN_solver_oneit.prototxt tmp_matlab_solver');
%     [~,m]=unix('echo "snapshot_prefix: \"SUN_matlab_solver\"" >> tmp_matlab_solver');
%     [~,m]=unix(['echo "max_iter: ' num2str(itteration+100) '" >> tmp_matlab_solver']);
%     [~,m]=unix('echo /media/martin/ssd-ext4/sun_test_dataset.h5 > matlab_dataset_test_list');
%     [~,m]=unix(['GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/caffe train -solver=tmp_matlab_solver -snapshot=SUN_matlab_solver_iter_' num2str(itteration) '.solverstate 2>> SUN_solver_oneit.prototxt.log']);
% end
% disp('done')
%exit

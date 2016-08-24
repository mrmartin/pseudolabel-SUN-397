if(false)
	%rewritten code for pseudolabel training with google images
	clear

	results_folder = 'pseudolabels_nogoogle_final'
	mkdir(results_folder)
	%% load data %%
	disp('loading labels and features')
	home = pwd;
	%load labels

	classes=dir('SUN_397_fc7/*.mat');
	classes=struct2cell(classes);    
	classes(2:end,:)=[];
	%% define my classes

	%load SUN dataset features
	sun_feat=single([]);
	sun_lab=single([]);
	for i=1:length(classes)
		tmp=load(['SUN_397_fc7/' classes{i}]);
		sun_feat=[sun_feat;single(tmp.sun_feat)];
		sun_lab=[sun_lab;i+zeros(size(tmp.sun_feat,1),1)];
	end
	clear tmp
	disp('loaded SUN dataset features');
	whos sun_feat sun_lab

	goo_feat=single([]);
	goo_lab=single([]);
	goo_locations=cell(0);
	counter=0;
	for i=1:length(classes)  
		file_loc=fopen(['google_397_fc7/' classes{i}(1:end-4) '_locations.txt']);
		tmp=textscan(file_loc,'%s');%use this information to reorder them
		bad_order=zeros(length(tmp{1}),1);
		for n=1:length(tmp{1})
			bad_order(n)=str2num(tmp{1}{n}(55+length(classes{i}):end-4));
		end
		[~, reorder]=sort(bad_order);

		for n=1:length(tmp{1})
			goo_locations{1}{n+counter}=tmp{1}{reorder(n)};
		end
		counter=length(goo_locations{1});
		fclose(file_loc);
		tmp=load(['/media/martin/MartinK3TB/Experiments/pseudolabel_SUN_397/google_397_fc7/' classes{i}]);
		goo_feat=[goo_feat;single(tmp.goo_feat(reorder,:))];
		goo_lab=[goo_lab;i+zeros(size(tmp.goo_feat,1),1)];
	end
	clear tmp counter
	disp('loaded GOOGLE features');
	whos goo_feat goo_lab

	disp(['number of SUN images per category is at least ' num2str(min(histc(sun_lab,unique(sun_lab))))])
	disp(['number of google images per category is at least ' num2str(min(histc(goo_lab,unique(goo_lab))))])
end

for diss_set_size=0%[20 100 500]%[1 5 50] %this is the proportion in each class! 10:50:450
	for min_belief=5%[5 1 2 10]%[2 5 15]
		for train_set_size=[20 5]%this is now set size!
			if((diss_set_size==0 && min_belief==5) || diss_set_size~=0)
				tic
				train_image_indexes=single([]);%unmixed indexes of sun_lab
				test_image_indexes=single([]);%unmixed indexes of sun_lab
				goo_image_indexes=single([]);%unmixed indexes of goo_lab

				disp(['sun_all_feats and sun_label are ordered. Let''s randomly select ' num2str(train_set_size) ' images from each SUN class for training and ' num2str(diss_set_size) ' images of each google class for dissolution'])
				rng default
				for i=1:length(classes)
					sun_indexes = find(sun_lab==i);%indexes of n images
					train_image_indexes = [train_image_indexes;sun_indexes(1:train_set_size)];%verified output
					test_image_indexes = [test_image_indexes;sun_indexes((train_set_size+1):end)];%verified output
				
					goo_indexes = find(goo_lab==i);%indexes of n images
					goo_image_indexes = [goo_image_indexes;goo_indexes(1:min(length(goo_indexes),diss_set_size))];%verified output%/100*histc(goo_lab,i)
				end
				clear goo_indexes sun_indexes
				spike=[];%randperm(length(test_image_indexes));

				%prepare test data
				test_order = randperm(length(test_image_indexes));
				delete /media/martin/ssd-ext4/sun_test_dataset.h5

				h5create('/media/martin/ssd-ext4/sun_test_dataset.h5','/data',[1 1 4096 length(test_order)],'DataType','single')
				h5create('/media/martin/ssd-ext4/sun_test_dataset.h5','/label',[1 length(test_order)],'DataType','single')
				h5write('/media/martin/ssd-ext4/sun_test_dataset.h5','/data',reshape(sun_feat(test_image_indexes(test_order),:)',[1,1,4096,length(test_order)]))
				h5write('/media/martin/ssd-ext4/sun_test_dataset.h5','/label',single(sun_lab(test_image_indexes(test_order))-1)')%INDEX FROM 0

				goo_training=[];
				goo_testing=randperm(length(goo_image_indexes));%mixed so that the selection accuracy can be tested in the first epoch

				[~,m]=unix('rm SUN_solver_oneit.prototxt.log');
				it_size=500;%has to be a multiple of 50
				start_from=0;%0
				clear chosen_accuracy chosen_number unused_accuracy goo_used_labels goo_used_label_beliefs
				for itteration=start_from:it_size:10000
					disp(['running itteration ' num2str(itteration)])
				
					%prepare training data
					disp(['writing new sun_dataset.h5, including ' num2str(length(goo_training)) ' pseudolabel images'])

					delete /media/martin/ssd-ext4/sun_dataset.h5
					h5create('/media/martin/ssd-ext4/sun_dataset.h5','/data',[1 1 4096 length(goo_training)+length(train_image_indexes)],'DataType','single')
					h5create('/media/martin/ssd-ext4/sun_dataset.h5','/label',[1 length(goo_training)+length(train_image_indexes)],'DataType','single')

					train_feat=[sun_feat(train_image_indexes,:);goo_feat(goo_image_indexes(goo_training),:)];%correct features, plus extra image for extra category
					train_lab=[sun_lab(train_image_indexes);goo_lab(goo_image_indexes(goo_training))];%add extra category
					whos('train_lab','train_feat')
			
					train_random=randperm(length(train_lab));					
					h5write('/media/martin/ssd-ext4/sun_dataset.h5','/data',reshape(single(train_feat(train_random,:)'),[1 1 4096 length(train_lab)]));
					h5write('/media/martin/ssd-ext4/sun_dataset.h5','/label',single(train_lab(train_random)-1)');
					
					clear train_feat train_lab;

					disp('training')
					[~,m]=unix('cp SUN_solver_oneit_train.prototxt tmp_matlab_solver');
					[~,m]=unix('echo "snapshot_prefix: \"SUN_matlab_solver\"" >> tmp_matlab_solver');
					[~,m]=unix(['echo "max_iter: ' num2str(itteration+it_size) '" >> tmp_matlab_solver']);
					[~,m]=unix(['echo "snapshot: ' num2str(it_size) '" >> tmp_matlab_solver']);
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

					if(diss_set_size~=0)
						if(false)
							disp('run evaluation on test images')
							[~,m]=unix('rm -r contaminated_perceptron_extracted');
							[~,m]=unix('echo /media/martin/ssd-ext4/sun_test_dataset.h5 > matlab_dataset_test_list');
							[~,mget]=unix(['GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/extract_features SUN_matlab_solver_iter_' num2str(itteration+it_size) '.caffemodel SUN_matlab_perceptron_extract.prototxt perceptron contaminated_perceptron_extracted ' num2str(ceil(length(test_image_indexes)/10))]);
							test_pred=csvread('contaminated_perceptron_extracted.csv');
							test_pred=test_pred(1:length(test_image_indexes),:);

							[a b]=max(test_pred,[],2);
							c=confusionmat(double(sun_lab(test_image_indexes(test_order))),b);
							test_class_accuracy=diag(c)./sum(c,2);
							%[test_class_accuracy test_class]=sort(diag(c)./sum(c,2));
							%get more images proportionally to how bad each class is doing
							goo_wanted_training=find(test_class_accuracy(goo_lab(goo_image_indexes))<rand(size(goo_image_indexes)));%naturally, if this works it needs to be chosen using train and unused accuracy!
						else
							goo_wanted_training=goo_image_indexes;
						end

						if(false)
							disp('rewriting unused goo set')
							delete /media/martin/ssd-ext4/sun_contaminated_dataset.h5
							h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',[1 1 4096 length(goo_testing)],'DataType','single')
							h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',[1 length(goo_testing)],'DataType','single')

							%mix and write goo for testing. No labels, no order randomization!
							h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',reshape(single(goo_feat(goo_image_indexes(goo_testing),:)'),[1 1 4096 length(goo_testing)]));
							h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',single(zeros(1,length(goo_testing))));

							disp('evaluating on contaminated test subset')
							[~,m]=unix('rm -r contaminated_perceptron_extracted');
							[~,m]=unix('echo /media/martin/ssd-ext4/sun_contaminated_dataset.h5 > matlab_dataset_test_list');
							[~,mget]=unix(['GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/extract_features SUN_matlab_solver_iter_' num2str(itteration+it_size) '.caffemodel SUN_matlab_perceptron_extract.prototxt perceptron contaminated_perceptron_extracted ' num2str(ceil(length(goo_testing)/10))]);

							unused_pred=csvread('contaminated_perceptron_extracted.csv');
							unused_pred=unused_pred(1:length(goo_testing),:);

							%select goo_training and goo_testing for next itteration
							[belief guess]=max(unused_pred,[],2);
							unused_true_labels=goo_lab(goo_image_indexes(goo_testing));
							[~,unused_true_indexes]=ismember(goo_image_indexes(goo_testing),goo_image_indexes);
							goo_used_labels(unused_true_indexes,itteration/it_size+1)=guess;
							goo_used_label_beliefs(unused_true_indexes,itteration/it_size+1)=belief;

							unused_accuracy(itteration/it_size+1)=sum(unused_true_labels==guess)/length(guess);

							%min belief
							%selected_unused=find(belief>min_belief);
							%max belief
							%selected_unused=find(belief<min_belief);
							%agreement
							%selected_unused=find(unused_true_labels==guess);
							%likely correct, not necessarily highest belief
							unused_true_belief=unused_pred(sub2ind(size(unused_pred),[1:length(unused_true_labels)]',double(unused_true_labels)));
							[highest_true_beliefs images_index]=sort(unused_true_belief,'descend');
							selected_unused=images_index(1:floor(length(goo_testing)/3));

							chosen_accuracy(itteration/it_size+1)=sum(unused_true_labels(selected_unused)==guess(selected_unused))/length(selected_unused);

							chosen_number(itteration/it_size+1)=length(selected_unused);

							%map the chosen images back to their index in goo_image_indexes
							[~,goo_training]=ismember(goo_image_indexes(goo_testing(selected_unused)),goo_image_indexes);
						
							if(chosen_accuracy(itteration/it_size+1)~=sum(goo_lab(goo_image_indexes(goo_training))==guess(selected_unused))/length(selected_unused))
								disp('the selection doesn''t add up, check it out!');
							end

							%goo_training = intersect(goo_wanted_training,goo_training);
						end
						goo_training=goo_training(rand(size(goo_training))>0.5);%randomly remove half
						
						goo_testing=setdiff(1:length(goo_image_indexes),goo_training);
					end
					[~,m]=unix('grep -B1 "accuracy = " SUN_solver_oneit.prototxt.log | grep -B1 Train | sed -e '':a'' -e ''N'' -e ''$!ba'' -e ''s/loss = [0-9\.]*\n//g'' | grep Iteration | sed ''s/.*Iteration \([0-9]*\).*accuracy = \([0-9\.]*\)/\1, \2/g'' > train_loss.csv');
					[~,m]=unix('grep "accuracy = " SUN_solver_oneit.prototxt.log | grep Test | sed ''s/.*accuracy = //g'' > test_loss.csv');
					train_accuracy=csvread('train_loss.csv');
					test_accuracy=csvread('test_loss.csv');
					test_accuracy=test_accuracy(1:length(train_accuracy),:);%crop to match train accuracy

					plot(train_accuracy(:,1),train_accuracy(:,2))
					hold on
					plot(train_accuracy(:,1),test_accuracy(1:end),'r')

					if(exist('chosen_number') && diss_set_size~=0)
						plot(train_accuracy((it_size/50):(it_size/50):length(unused_accuracy)*(it_size/50),1),unused_accuracy,'k')

						plot(train_accuracy((it_size/50):(it_size/50):length(chosen_accuracy)*(it_size/50),1),chosen_accuracy,'g')
						text(train_accuracy((it_size/50):(it_size/50):length(chosen_accuracy)*(it_size/50),1),chosen_accuracy,num2str(chosen_number'),'horiz','center','vert','bottom')
					end

					legend('training accuracy','test accuracy',['contaminated accuracy (' num2str(length(goo_image_indexes)) ' images)'],['contaminated accuracy which are chosen'],'Location','northwest')
					ylim([0 1])
					hold off
					title([num2str(train_set_size) ' training images and ' num2str(diss_set_size) ' top images from google'])%, with max belief ' num2str(min_belief)])
					xlabel('iterations')
					ylabel('accuracy')
					pause(3)
					toc
				end
				if(exist('chosen_number'))
					save([results_folder '/accuracy_' num2str(train_set_size) '_' num2str(diss_set_size) '_' num2str(min_belief)],'train_accuracy','test_accuracy','chosen_number','chosen_accuracy','goo_used_labels','goo_used_label_beliefs')
				else
					save([results_folder '/accuracy_' num2str(train_set_size) '_' num2str(diss_set_size) '_' num2str(min_belief)],'train_accuracy','test_accuracy')
				end
				print('-f1','-dpng',[results_folder '/accuracy_' num2str(train_set_size) '_' num2str(diss_set_size) '_' num2str(min_belief) '.png'])
			end
		end
	end
end
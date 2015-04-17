if(~exist('test_set_proportion'))
    test_set_proportion=0.7;
end

%you need lots of swap to hold this, make sure the SSD is connected!
clearvars -except test_set_proportion

test_set_proportion

results_folder = '10_to_450_test_over_10_to_450_diss_one_class'
mkdir(results_folder)
current_class = 1

%% load data %%
disp('loading data')
home = pwd;
cd '../../Datasets/SUN397/SUN_6_scenes/227x227/4096'
%load data
preds=dir('*_pred');
image_data=struct([]);

all_features=[];
all_labels=[];
for i=[current_class length(preds)]
   tic;disp(['loading ' preds(i).name]);
   thisone=csvread(preds(i).name);
   image_data(i).features=thisone;
   image_data(i).labels=i*ones(size(thisone,1),1);

   all_features=[all_features;thisone];
   all_labels=[all_labels;i*ones(size(thisone,1),1)];
   toc
   %pause(1)%gives matlab time to display size of current variables in memory
end

cd(home);
clear thisone
%tic;load /media/martin/ssd-ext4/features_and_labels;toc

for diss_set_size=10:100:450
    for train_set_size=10:100:450    
        %% define my classes
        classes={'airport_terminal','bedroom','castle','dining','kitchen','living_room','other'}'
        for i=[current_class length(preds)]
            fprintf('%d - %s\n',histc(all_labels,i),classes{i});
        end
        disp(['number of images per category is at least ' num2str(min(histc(all_labels,unique(all_labels))))])

        image_orders=struct([]);

        disp(['so let''s take 400 for training, 400 for dissolution, and the rest for testing'])
        for i=[current_class]
            rng default
            set_order = randperm(histc(all_labels,i));
            image_orders(i).train_set = set_order(1:train_set_size);
            image_orders(i).diss_set = set_order((train_set_size+1):(train_set_size+diss_set_size));
            image_orders(i).test_set = set_order((train_set_size+diss_set_size+1):end);
        end
        image_orders

        train_features=[];
        train_labels=[];

        for i=[current_class]
            train_labels=[train_labels;image_data(i).labels(image_orders(i).train_set)];
            train_features=[train_features;image_data(i).features(image_orders(i).train_set,:)];
        end

        training_positive = floor(length(train_labels)/2);

        disp(['"other" images are required in the training data too. Let''s use ' num2str(training_positive)])
        rng default
        other_order = randperm(length(image_data(length(classes)).labels));

        train_features=[train_features;image_data(length(classes)).features(other_order(1:training_positive),:)];
        train_labels=[train_labels;image_data(length(classes)).labels(other_order(1:training_positive))];

        diss_features=[];
        diss_labels=[];

        for i=[current_class]
            diss_labels=[diss_labels;image_data(i).labels(image_orders(i).diss_set)];
            diss_features=[diss_features;image_data(i).features(image_orders(i).diss_set,:)];
        end

        diss_positive=250;%floor(length(diss_labels)/2);

        disp(['"other" images are required in the contaminated data too. Let''s use ' num2str(diss_positive)])

        diss_features=[diss_features;image_data(length(classes)).features(other_order(training_positive+(1:diss_positive)),:)];
        diss_labels=[diss_labels;image_data(length(classes)).labels(other_order(training_positive+(1:diss_positive)))];

        test_features=[];
        test_labels=[];

        for i=[current_class]
            test_labels=[test_labels;image_data(i).labels(image_orders(i).test_set)];
            test_features=[test_features;image_data(i).features(image_orders(i).test_set,:)];
        end

        disp('all "other" images are used as negative, and in the contaminated data, but not in test set')
        rng default
        neg_order = randperm(length(image_data(7).labels));
        neg_features = image_data(7).features(neg_order,:);
        neg_labels = image_data(7).labels(neg_order,:);

        clear ans i set_order

        %% save data for itteration 0, where no contaminated data are used in training
        rng default
        disp(['before saving to HDF5, here''s a random number: ' num2str(rand)]) %print one random number, so that random oprders below here are different from before
        train_order = randperm(length(train_labels));
        delete /media/martin/ssd-ext4/sun_dataset.h5

        %prepare h5 dataset file, and fill it with data
        h5create('/media/martin/ssd-ext4/sun_dataset.h5','/data',[1 1 4096 length(train_labels)],'DataType','single')
        h5create('/media/martin/ssd-ext4/sun_dataset.h5','/label',[1 length(train_labels)],'DataType','single')

        h5write('/media/martin/ssd-ext4/sun_dataset.h5','/data',single(reshape(train_features(train_order,:)',[1,1,4096,length(train_labels)])))
        h5write('/media/martin/ssd-ext4/sun_dataset.h5','/label',single(train_labels(train_order)-1)')%INDEX FROM 0

        %diss_order = randperm(length(diss_labels));
        delete /media/martin/ssd-ext4/sun_contaminated_dataset.h5

        %prepare h5 dataset file, and fill it with data
        h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',[1 1 4096 length(diss_labels)],'DataType','single')
        h5create('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',[1 length(diss_labels)],'DataType','single')

        h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/data',single(reshape(diss_features',[1,1,4096,length(diss_labels)])))
        h5write('/media/martin/ssd-ext4/sun_contaminated_dataset.h5','/label',single(diss_labels-1)')%INDEX FROM 0

        %test_order = randperm(length(test_labels));
        delete /media/martin/ssd-ext4/sun_test_dataset.h5

        %prepare h5 dataset file, and fill it with data
        h5create('/media/martin/ssd-ext4/sun_test_dataset.h5','/data',[1 1 4096 length(test_labels)],'DataType','single')
        h5create('/media/martin/ssd-ext4/sun_test_dataset.h5','/label',[1 length(test_labels)],'DataType','single')

        h5write('/media/martin/ssd-ext4/sun_test_dataset.h5','/data',single(reshape(test_features',[1,1,4096,length(test_labels)])))
        h5write('/media/martin/ssd-ext4/sun_test_dataset.h5','/label',single(test_labels-1)')%INDEX FROM 0

        %% run one itteration of caffe
        [~,m]=unix('rm SUN_solver_oneit.prototxt.log');
        it_size=100;
        for itteration=0:it_size:50000
            disp(['running itteration ' num2str(itteration)])

            [~,m]=unix('cp SUN_solver_oneit_train.prototxt tmp_matlab_solver');
            [~,m]=unix('echo "snapshot_prefix: \"SUN_matlab_solver\"" >> tmp_matlab_solver');
            [~,m]=unix(['echo "max_iter: ' num2str(itteration+it_size) '" >> tmp_matlab_solver']);
            [~,m]=unix('echo /media/martin/ssd-ext4/sun_test_dataset.h5 > matlab_dataset_test_list');
            if(itteration==0)
                [~,m]=unix('GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/caffe train -solver=tmp_matlab_solver 2>> SUN_solver_oneit.prototxt.log');
            else
                [~,m]=unix(['GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/caffe train -solver=tmp_matlab_solver -snapshot=SUN_matlab_solver_iter_' num2str(itteration) '.solverstate 2>> SUN_solver_oneit.prototxt.log']);
            end
            %evaluate on diss_set
            [~,m]=unix('rm -r contaminated_perceptron_extracted');
            [~,m]=unix('echo /media/martin/ssd-ext4/sun_contaminated_dataset.h5 > matlab_dataset_test_list');
            [~,mget]=unix(['GLOG_logtostderr=1 /media/martin/MartinK3TB/Documents/caffe/build/tools/extract_features SUN_matlab_solver_iter_' num2str(itteration+it_size) '.caffemodel SUN_matlab_perceptron_extract.prototxt perceptron contaminated_perceptron_extracted ' num2str(ceil(length(diss_labels)/10))]);
            %
            %for each class take the most likely from the diss set and put them in the
            %train set. These will be changed in the next itteration
            diss_pred=csvread('contaminated_perceptron_extracted.csv');
            %how correct is it? (the learning process doesn't know this, but it's
            %interesting for analysis)
            [a,b]=max(diss_pred,[],2);
            c=confusionmat(diss_labels,b);
            fprintf('%d\t%d\t%s\n',c(1,:),classes{current_class})
            fprintf('%d\t%d\t%s\n',c(2,:),classes{7})
            %pick a random subset of the most probable
            rand_diss_set=randperm(length(b));
            rand_diss_set=rand_diss_set(1:length(b)/4);

            oneit_train_labels=[train_labels;b(rand_diss_set)];
            oneit_train_features=[train_features;diss_features(rand_diss_set,:)];

            train_order = randperm(length(oneit_train_labels));
            delete /media/martin/ssd-ext4/sun_dataset.h5

            %prepare h5 dataset file, and fill it with data
            h5create('/media/martin/ssd-ext4/sun_dataset.h5','/data',[1 1 4096 length(oneit_train_labels)],'DataType','single')
            h5create('/media/martin/ssd-ext4/sun_dataset.h5','/label',[1 length(oneit_train_labels)],'DataType','single')

            h5write('/media/martin/ssd-ext4/sun_dataset.h5','/data',single(reshape(oneit_train_features(train_order,:)',[1,1,4096,length(oneit_train_labels)])))
            h5write('/media/martin/ssd-ext4/sun_dataset.h5','/label',single(oneit_train_labels(train_order)-1)')%INDEX FROM 0

            %[~,m]=unix('grep loss SUN_solver_oneit.prototxt.log | grep Iteration | sed -n ''n;p'' | sed ''s/^.*Iteration \([0-9]*\), loss = \([0-9\.]*\)$/\1, \2/g'' > train_loss.csv');
            [~,m]=unix('grep -B1 "accuracy = " SUN_solver_oneit.prototxt.log | grep -B1 Train | sed -e '':a'' -e ''N'' -e ''$!ba'' -e ''s/loss = [0-9\.]*\n//g'' | grep Iteration | sed ''s/.*Iteration \([0-9]*\).*accuracy = \([0-9\.]*\)/\1, \2/g'' > train_loss.csv');
            %[~,m]=unix('grep loss SUN_solver_oneit.prototxt.log | grep Iteration | sed -n ''p;n'' | sed ''s/^.*Iteration \([0-9]*\), loss = \([0-9\.]*\)$/\1, \2/g'' > test_loss.csv');
            [~,m]=unix('grep "accuracy = " SUN_solver_oneit.prototxt.log | grep Test | sed ''s/.*accuracy = //g'' > test_loss.csv');
            train_accuracy=csvread('train_loss.csv');
            test_accuracy=csvread('test_loss.csv');

            plot(train_accuracy(:,1),train_accuracy(:,2))
            hold on
            plot(train_accuracy(:,1),test_accuracy(1:2:(2*size(train_accuracy,1))),'r')
            legend('training accuracy','test accuracy','Location','southeast')
            ylim([0 1])
            hold off
            title([num2str(train_set_size) classes{current_class} ' and ' num2str(diss_set_size) ' dissolved in ' num2str(length(diss_labels)) ' contaminated images'])
            xlabel('iterations')
            ylabel('accuracy')
            pause(0.1)
        end
        save([results_folder '/accuracy_' num2str(train_set_size) '_' num2str(diss_set_size) '_class_' num2str(current_class)],'train_accuracy','test_accuracy')
        print('-f1','-dpng',[results_folder '/accuracy_' num2str(train_set_size) '_' num2str(diss_set_size) '_class_' num2str(current_class) '.png'])
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
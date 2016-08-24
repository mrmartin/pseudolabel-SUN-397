clear
home=pwd;
results_folder = 'pseudolabels_belief_threshold_final'%'10_to_450_test_over_10_to_450_diss_separate_pseudolabels'
cd(results_folder)

%load each set of results into a single 3D matrix
files=dir('*.mat');
%files(4)=[];
clear accuracies

expected_samples=3*3;
tmp=load(files(2).name);
number_of_itterations=length(tmp.train_accuracy(:,1));
%labels=zeros(expected_samples,2);
%accuracies=zeros(expected_samples,3,number_of_itterations);
for i=1:length(files)
    tmp=strsplit(files(i).name(10:end-4),'_');
    labels(i,1)=str2num(tmp{1});
    labels(i,2)=str2num(tmp{2});
    tmp=load(files(i).name);
    accuracies(i,1,:)=tmp.train_accuracy(1:number_of_itterations,1);%itterations
    accuracies(i,2,:)=tmp.train_accuracy(1:number_of_itterations,2);%train accuracy
    accuracies(i,3,:)=tmp.test_accuracy(1:number_of_itterations);%test_accuracy
end
cd(home)%labels

%rescramble so that labels are in order
rearranged_train_accuracies=zeros(length(unique(labels(:,1))),length(unique(labels(:,2))),number_of_itterations);
rearranged_test_accuracies=zeros(length(unique(labels(:,1))),length(unique(labels(:,2))),number_of_itterations);
for train_label=unique(labels(:,1))'
    for diss_label=unique(labels(:,2))'
        if(sum(labels(:,1)==train_label & labels(:,2)==diss_label)>0)
            rearranged_train_accuracies(unique(labels(:,2))==diss_label,unique(labels(:,1))==train_label,:)=accuracies(labels(:,1)==train_label & labels(:,2)==diss_label,2,:);
            rearranged_test_accuracies(unique(labels(:,2))==diss_label,unique(labels(:,1))==train_label,:)=accuracies(labels(:,1)==train_label & labels(:,2)==diss_label,3,:);
        end
    end
end

%%
%set(gcf, 'Position', [0, 0, 1100, 400]);
average_accuracy_length=5;

%won't work if diss resolution != train resolution:
resolution=sqrt(expected_samples);%this changes for diss_set_size=10:100:450 and train_set_size=10:100:450 

%final_train_accuracy=reshape(accuracies(:,2,end),resolution,resolution);
mean_train_accuracy=[999 793 547;945 682 542;810 651 541;604 580 531]/1000%mean(rearranged_train_accuracies(:,:,end-average_accuracy_length:end),3);%reshape(mean(accuracies(:,2,end-100:end),3),resolution,resolution);

subplot(1,3,1)
imagesc(mean_train_accuracy)
colormap(gray);
%caxis([0 1]);
title('train accuracy')
set(gca,'XTick',[1:length(unique(labels(:,1)))])
set(gca,'XTickLabel',unique(labels(:,1)))
xlabel('# training images')
set(gca,'YTick',[1:4])
set(gca,'YTickLabel',[0 20 100 500])
ylabel('# google images')
 
textStrings = num2str(mean_train_accuracy(:),'%0.3f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:4);   %# Create x and y coordinates for the strings
x(:,end)=[];
y(:,end)=[];
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mean_train_accuracy(:) < midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

%final_test_accuracy=reshape(accuracies(:,3,end),resolution,resolution);
mean_test_accuracy=[248 362 393;401 475 508;397 479 510; 369 441 487]/1000;%mean(rearranged_test_accuracies(:,:,end-average_accuracy_length:end),3);%mean_test_accuracy=reshape(mean(accuracies(:,3,end-100:end),3),resolution,resolution);
subplot(1,3,2)
imagesc(mean_test_accuracy)
colormap(gray);
%caxis([0 1]);
title('test accuracy')
set(gca,'XTick',[1:length(unique(labels(:,1)))])
set(gca,'XTickLabel',unique(labels(:,1)))
xlabel('# training images')
set(gca,'YTick',[1:4])
set(gca,'YTickLabel',[0 20 100 500])
ylabel('# google images')
% 
textStrings = num2str(mean_test_accuracy(:),'%0.3f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:4);   %# Create x and y coordinates for the strings
x(:,end)=[];
y(:,end)=[];
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');
midValue = 0.4;%mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mean_test_accuracy(:) < midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

%final_test_accuracy=reshape(accuracies(:,3,end),resolution,resolution);
mean_test_accuracy=[0 0 0;950 837 798;846 770 715;676 659 626]/1000;%mean(rearranged_test_accuracies(:,:,end-average_accuracy_length:end),3);%mean_test_accuracy=reshape(mean(accuracies(:,3,end-100:end),3),resolution,resolution);
subplot(1,3,3)
imagesc(mean_test_accuracy)
colormap(gray);
%caxis([0 1]);
title('pseudolabel accuracy')
set(gca,'XTick',[1:length(unique(labels(:,1)))])
set(gca,'XTickLabel',unique(labels(:,1)))
xlabel('# training images')
set(gca,'YTick',[1:4])
set(gca,'YTickLabel',[0 20 100 500])
ylabel('# google images')
% 
textStrings = num2str(mean_test_accuracy(:),'%0.3f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:4);   %# Create x and y coordinates for the strings
x(:,end)=[];
y(:,end)=[];
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');
midValue = 0.4;%mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mean_test_accuracy(:) < midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

subplot(1,3,1)
imagesc(mean_train_accuracy)
colormap(gray);
%caxis([0 1]);
title('train accuracy')
set(gca,'XTick',[1:length(unique(labels(:,1)))])
set(gca,'XTickLabel',unique(labels(:,1)))
xlabel('# training images')
set(gca,'YTick',[1:4])
set(gca,'YTickLabel',[0 20 100 500])
ylabel('# google images')
 
textStrings = num2str(mean_train_accuracy(:),'%0.3f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:4);   %# Create x and y coordinates for the strings
x(:,end)=[];
y(:,end)=[];
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mean_train_accuracy(:) < midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

results_folder(results_folder=='_')=' ';
%suptitle(results_folder)

disp('The ratio of false negatives to false positives, each normalized across the number of images, quantifies the quality of the negative set')
disp('Experimental results demonstrate that this method is of mainly of benefit when the number of training samples is small (20 - 100).')
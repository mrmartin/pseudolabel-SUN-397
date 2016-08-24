if(~exist('goo_lab'))
	classes=dir('SUN_397_fc7/*.mat');
	classes=struct2cell(classes);    
	classes(2:end,:)=[];

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
		goo_lab=[goo_lab;i+zeros(size(tmp.goo_feat,1),1)];
	end
end

%confusion between mode predictions and reality
conf=confusionmat(mode(goo_used_labels(:,3:end)')',double(goo_lab(goo_image_indexes)));
conf=conf(2:end,2:end);%remove zeros

%in confusionmat, the first index is for the left, second index for the right

%find most occuring confusions
conf=conf-conf.*eye(397);%remove diagonal
[a worst]=sort(conf(:),'descend');
for i=1:10
	[c d]=ind2sub([397 397],worst(i))
	%c is what CNN says, d is what google says
	disp([num2str(a(i)) ' times, CNN said ' classes{c}(1:end-4) ', but google had retrieved ' classes{d}(1:end-4)])
end

n=randperm(length(goo_image_indexes));
for i=n(1:100)
	imagesc(imread(goo_locations{goo_image_indexes(i)}))
	belief_in_class = goo_used_label_beliefs(i,goo_used_labels(i,11)==goo_used_labels(i,:))
	title(['google: ' classes{goo_lab(goo_image_indexes(i))}(1:end-4) ', CNN: ' classes{max(1,goo_used_labels(i,end))}(1:end-4) '(' num2str(sum(belief_in_class)) ')'])
	goo_used_labels(i,:)
	pause 
end  
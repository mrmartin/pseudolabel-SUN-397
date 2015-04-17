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
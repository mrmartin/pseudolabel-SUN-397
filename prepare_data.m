goo=dir('google_397_fc7/*.csv');
sun=dir('SUN_397_fc7/*.csv');
goo_feat=[];
sun_feat=[];
goo_label=[];
sun_label=[];
for i=1:length(goo)
goo_feat=csvread(['google_397_fc7/' goo(i).name]);
save(['google_397_fc7/' goo(i).name(5:end-4)],'goo_feat');
%goo_label=[goo_label;zeros(size(goo_tmp,1),1)+i];
%goo_feat=[goo_feat;goo_tmp];
sun_feat=csvread(['SUN_397_fc7/' sun(i).name]);
save(['SUN_397_fc7/' sun(i).name(5:end-4)],'sun_feat');
%sun_label=[sun_label;zeros(size(sun_tmp,1),1)+i];
%sun_feat=[sun_feat;sun_tmp];
disp(['google_397_fc7/' goo(i).name(5:end-4) ' matches SUN_397_fc7/' sun(i).name(5:end-4)]);
end


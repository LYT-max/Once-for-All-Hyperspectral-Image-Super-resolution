clear all
close all
clc 

% Settings
ratio       = 8;
kernel_size = [8, 8];
sig         = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1;
start_pos(2)=1;

%Creating patches
patch_size = 512;
n_spectral = 128;
H_patch_size = patch_size;

size_label =patch_size;
stride = 512;

%% initialization
label = zeros(size_label, size_label, n_spectral, 1,1);
LR = zeros(H_patch_size, H_patch_size, n_spectral, 1,1);

count=0;
tic 
 
    load("chikusei_crop.mat");
    orig = chikusei_crop(1:512, 1:2048, :);
    orig = orig/max(orig(:));
    [L, W, C] = size(orig);
    
 
    
    for l = 1 : stride :L-size_label+1
        for w = 1 :stride : W-size_label+1
            
            ref = orig(l : l+size_label-1, w :w+size_label-1,:); 
            [y, BC] = downsample(ref,ratio,kernel_size,sig,start_pos);
           
	        y = imresize(y, ratio, 'bicubic');
           
            
            count=count+1;
   
     
            label(:, :, :,count) = ref;
            LR(:, :, :,count) = y;
            
   
        end
    end
    
%save 'train_16_64' label PAN MSI LR;

figure, imshow(LR(:, :,[70, 100, 36],1));
figure, imshow(label(:, :,[70, 100, 36],1));


savepath = "./test_chikusei_512_512_scale_8.h5";

order = [1:4];
dataa = LR(:, :, :, order);
label = label(:, :, :, order); 
chunksz = 4;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
   last_read=(batchno-1)*chunksz;
   batchdata = dataa(:,:,:,last_read+1:last_read+chunksz); 
   batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

   
   disp([batchno, floor(count/chunksz)])
   startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
   curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
   created_flag = true;
   totalct = curr_dat_sz(end);
end
toc
h5disp(savepath);



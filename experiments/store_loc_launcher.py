from .tmux_launcher import Options, TmuxLauncher
from datetime import datetime
from os.path import join

class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="E:\\Personal_Stuff\\UIUC\\Sem_2\\DL_for_CV_(CS_444)\\Project\\Code",#dataroot
                name="DL_for_CV",#experiment name
                CUT_mode="FastCUT",# CUT/Fast_CUT
                gpu_ids="0", #Can provide multiple GPUs for distributed training
                #checkpoints_dir= join("work_dir",datetime.now().strftime("%Y%m%d-%H%M%S")),
                checkpoints_dir="\work_dir_cut",
                model="cut",#https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/__init__.py#L25
                netD="basic",
                netG="resnet_9blocks",
                normG="instance",
                normD="instance",
                dataset_mode="unaligned",
                direction="AtoB",
                num_threads=4,
                batch_size=1, #default=1 works best for them
                load_size=512, #since preprocess=none, this does not do anything
                crop_size=512, #since preprocess=none, this does not do anything
                preprocess="resize_to_half", #r1, scale_shortside_and_zoom, none#resize_to_half
                verbose="",
                #train options`
                # print_freq=100,
                # log_freq=100,
                # save_latest_freq=5000,
                # save_epoch_freq=100,
                # eval_epoch_freq=100,
                phase="test",
                # n_epochs=50,
                # n_epochs_decay=30,
                # lr=0.0002,
                # gan_mode="lsgan",
                # lr_policy="linear",
                results_dir="\work_dir_cut",
                isTrain="",
                num_test_images=1,
                short_side_thresh=512,
                long_side_thresh=512,
                dirA="Inria\\train\\gt",#Store/train syn_combo_datasets/images inria
                dirB="WHU\\image",# IS_real_data/LOC_store/train_images whu
                lambda_NCE=2.0
                #continue_train="",
                #num_patches=512,
                #netF_nc=512,  # Dimension of MLP layer
            ),
            Options(
                checkpoints_dir="E:\\work_dir_cut",
                dataset_mode="single",
                dataroot="/home/virajns2",  # dataroot
                name="store_to_loc_CUT",  # experiment name
                CUT_mode="Fast_CUT",  # CUT/Fast_CUT
                gpu_ids="1",  # Can provide multiple GPUs for distributed training
                results_dir="/Desktop",
                phase="test",
                preprocess="none",  # r1
                dirA="Store/test",
                dirB="LOC",
                save_image_prefix_len=1,
                num_test=10000,
                epoch="1",
                short_side_thresh=512,
                long_side_thresh=768
            ),
            #TEST ONLY A using SINGLE_DATASET
            Options(
                checkpoints_dir="/home/virajns2/work_dir",
                dataset_mode="single",
                dataroot="/home/jupyter/data/Duality/Prod_Rec_real_data/CUT_data",  # dataroot
                name="store_to_loc_CUT",  # experiment name
                CUT_mode="CUT",  # CUT/Fast_CUT
                gpu_ids="1",  # Can provide multiple GPUs for distributed training
                results_dir="/home/jupyter/model_logs/CUT/20220902-064409/test_results_only_A/",
                phase="test",
                preprocess="none",  # none#resize_to_half
                dirA="Store/test",
                dirB="LOC",
                save_image_prefix_len=1,
                num_test=10000,
                epoch="latest"
            ),
            #lambda NCE
            Options(
                dataroot="/home/virajns2",  # dataroot
                name="store_to_loc_CUT",  # experiment name
                CUT_mode="Fast_CUT",  # CUT/Fast_CUT
                gpu_ids="1",  # Can provide multiple GPUs for distributed training
                #checkpoints_dir=join("/home/jupyter/model_logs/CUT", datetime.now().strftime("%Y%m%d-%H%M%S")),
                checkpoints_dir="/home/virajns2/work_dir",
                model="cut",
                # https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/__init__.py#L25
                netD="basic",
                netG="resnet_9blocks",
                normG="instance",
                normD="instance",
                dataset_mode="unaligned",
                direction="AtoB",
                num_threads=4,
                batch_size=1,  # default=1 works best for them
                load_size=768,  # since preprocess=none, this does not do anything
                crop_size=256,  # since preprocess=none, this does not do anything
                preprocess="r1",  # none#resize_to_half
                verbose="",
                # train options`
                print_freq=100,
                log_freq=100,
                save_latest_freq=5000,
                save_epoch_freq=1,
                eval_epoch_freq=1,
                phase="train",
                n_epochs=50,
                n_epochs_decay=30,
                lr=0.0002,
                gan_mode="lsgan",
                lr_policy="linear",
                isTrain="",
                num_test_images=500,
                short_side_thresh=320,
                long_side_thresh=320,
                dirA="Store/train",
                dirB="LOC_train",
                num_patches=512,
                netF_nc=512,# Dimension of MLP layer
                lambda_NCE=2.0,
                continue_train=""
                # continue_train="",
            ),
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        return ["python test.py " + str(opt) for opt in self.common_options()]

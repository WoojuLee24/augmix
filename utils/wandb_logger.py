from utils import plot_confusion_matrix
import pdb
import pandas as pd
import json

class WandbLogger():
    def __init__(self,
                 config,
                 args = None,
                 wandb_init_kwargs=None,
                 interval=10,
                 log_map_every_iter=True,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 **kwargs):
        if config is None:
            self.use_wandb = False
        else:
            self.use_wandb = True
            self.config = config
            self.args = args
            self.import_wandb()

            if self.args.dataset == 'imagenet':
                with open("/ws/data/imagenet/imagenet_class_index.json", "r") as json_file:
                    class_dict = json.load(json_file)
                self.class_list = []
                for key, value in class_dict.items():
                    self.class_list.append(value[1])
            else:
                self.class_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def log(self, key, value):
        self.wandb.log({key: value})


    def convert_to_df(self, wandb_input):
        df = pd.DataFrame(wandb_input)
        df = df.set_axis(labels=self.class_list, axis=0)
        df = df.set_axis(labels=self.class_list, axis=1)
        df_wandb = self.wandb.Table(data=df)
        return df_wandb

    ###########
    ### run ###
    ###########
    def before_run(self):
        if self.use_wandb:
            if self.wandb is None:
                self.import_wandb()
            if self.config:
                self.wandb.init(**self.config)
            else:
                self.wandb.init()

    def after_run(self, wandb_input):
        if self.use_wandb:
            if 'test_c_table' in wandb_input:
                test_c_table = self.wandb.Table(data=wandb_input['test_c_table'])
                self.wandb.log({"test_c_results": test_c_table})
                # for key, value in test_c_features.items():
                #     wandb.log({key: value})
            if 'test_c_acc' in wandb_input:
                self.wandb.log({"test/corruption_error: ": 100 - 100. * wandb_input['test_c_acc']})
            if 'test_c_cm' in wandb_input:
                if self.args.dataset == 'imagenet':
                    # test_c_cm = self.convert_to_df(wandb_input['test_c_cm'])
                    # self.wandb.log({'test_c_cm': test_c_cm})
                    pass
                else:
                    test_c_plt = plot_confusion_matrix(wandb_input['test_c_cm'])
                    self.wandb.log({'clean': test_c_plt})

            self.wandb.finish()

    #############
    ### epoch ###
    #############
    def before_train_epoch(self):
        if self.use_wandb:
            pass

    def after_train_epoch(self, wandb_input):
        if self.use_wandb:
            # log wandb features
            if 'train_features' in wandb_input:
                for key, value in wandb_input['train_features'].items():
                    self.wandb.log({key: value})

    def after_test_epoch(self, wandb_input):
        if self.use_wandb:
            if 'test_features' in wandb_input:
                for key, value in wandb_input['test_features'].items():
                    self.wandb.log({key: value})
            if 'test_cm' in wandb_input:
                if self.args.dataset == 'imagenet':
                    # test_cm = self.convert_to_df(wandb_input['test_cm'])
                    # self.wandb.log({'test_cm': test_cm})
                    pass
                else:
                    test_plt = plot_confusion_matrix(wandb_input['test_cm'])
                    self.wandb.log({'clean': test_plt})

    ############
    ### iter ###
    ############
    def before_train_iter(self):
        if self.use_wandb:
            pass

    def after_train_iter(self, wandb_input):
        if self.use_wandb:
            if 'tsne_features' in wandb_input:
                self.wandb.log({f"t-sne(features)": self.wandb.Image(wandb_input['tsne_features'])})
            if 'tsne_logits' in wandb_input:
                self.wandb.log({f"t-sne(logits)": self.wandb.Image(wandb_input['tsne_logits'])})
            if 'tsne' in wandb_input:
                self.wandb.log({f"t-sne": self.wandb.Image(wandb_input['tsne'])})

            log_list = ['loss', 'acc1', 'acc5']
            for log_item in log_list:
                if log_item in wandb_input:
                    self.wandb.log({f'{log_item}': wandb_input[log_item]})

    ###########
    ### ETC ###
    ###########
    def log_evaluate(self, wandb_input):
        if self.use_wandb:
            # log wandb features
            if 'train_features' in wandb_input:
                for key, value in wandb_input['train_features'].items():
                    self.wandb.log({key: value})
            if 'test_features' in wandb_input:
                for key, value in wandb_input['test_features'].items():
                    self.wandb.log({key: value})
            if 'test_c_features' in wandb_input:
                for key, value in wandb_input['test_c_features'].items():
                    self.wandb.log({key: value})
            if 'test_acc' in wandb_input:
                self.wandb.log({"test/clean_error: ": 100 - 100. * wandb_input['test_acc']})
            if 'test_cm' in wandb_input:
                if self.args.dataset == 'imagenet':
                    # test_cm = self.convert_to_df(wandb_input['test_cm'])
                    # self.wandb.log({'test_cm': test_cm})
                    pass
                else:
                    test_plt = plot_confusion_matrix(wandb_input['test_cm'])
                    self.wandb.log({'test/clean': test_plt})
            if 'test_c_table' in wandb_input:
                test_c_table = self.wandb.Table(data=wandb_input['test_c_table'])
                self.wandb.log({"test_c_results": test_c_table})
            if 'test_c_error' in wandb_input:
                self.wandb.log({"test/corruption_error: ": wandb_input['test_c_error']})
            if 'test_c_acc' in wandb_input:
                self.wandb.log({"test/corruption_error: ": 100 - 100. * wandb_input['test_c_acc']})
            if 'test_c_cm' in wandb_input:
                if self.args.dataset == 'imagenet':
                    # test_c_cm = self.convert_to_df(wandb_input['test_c_cm'])
                    # self.wandb.log({'test_c_cm': test_c_cm})
                    pass
                else:
                    test_c_plt = plot_confusion_matrix(wandb_input['test_c_cm'])
                    self.wandb.log({'corruption': test_c_plt})
            if 'test_c_cms' in wandb_input:
                if self.args.dataset == 'imagenet':
                    # for key, value in wandb_input['test_c_cms'].items():
                    #     test_c_cm = self.convert_to_df(value)
                    #     self.wandb.log({key: test_c_cm})
                    pass
                else:
                    for key, value in wandb_input['test_c_cms'].items():
                        test_c_plt = plot_confusion_matrix(value)
                        self.wandb.log({key: test_c_plt})
            if 'train_cms' in wandb_input:
                if self.args.dataset == 'imagenet':
                    # for key, value in wandb_input['train_cms'].items():
                    #     train_cm = self.convert_to_df(value)
                    #     self.wandb.log({key: train_cm})
                    pass
                else:
                    for key, value in wandb_input['train_cms'].items():
                        train_plt = plot_confusion_matrix(value)
                        self.wandb.log({key: train_plt})

            # # 'tsne plot -> no figure saved: debug required'
            # if 'tsne_features' in wandb_input:
            #     self.wandb.log({f"t-sne(features)": self.wandb.Image(wandb_input['tsne_features'])})
            # if 'tsne_logits' in wandb_input:
            #     self.wandb.log({f"t-sne(logits)": self.wandb.Image(wandb_input['tsne_logits'])})
            # if 'tsne' in wandb_input:
            #     self.wandb.log({f"t-sne": self.wandb.Image(wandb_input['tsne'])})
            # if 'test_c_tsne' in wandb_input:
            #     for key, tsne in wandb_input['test_c_tsne'].items():
            #         # pdb.set_trace()
            #         # tsne.savefig('/ws/data/debug/debug2.jpg')
            #         self.wandb.log({f"t-sne": self.wandb.Image(tsne)})

            # for key, value in test_c_features.items():
            #     self.wandb.log({key: value})
            # test_c_plt = plot_confusion_matrix(test_c_cm)
            # for key, value in test_c_plt.items():
            #     self.wandb.log({key: value})

    def log_analysis(self, wandb_input):
        if self.use_wandb:
            if 'test_dg_table' in wandb_input:
                test_dg_table = self.wandb.Table(data=wandb_input['test_dg_table'])
                self.wandb.log({"additional_loss": test_dg_table})
            if 'test_dg_features' in wandb_input:
                for key, value in wandb_input['test_dg_features'].items():
                    self.wandb.log({key: value})
from utils import plot_confusion_matrix
import pdb

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

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def log(self, key, value):
        self.wandb.log({key: value})

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
            if 'test_cm' in wandb_input:
                test_plt = plot_confusion_matrix(wandb_input['test_cm'])
                self.wandb.log({'clean': test_plt})
            if 'test_c_table' in wandb_input:
                test_c_table = self.wandb.Table(data=wandb_input['test_c_table'])
                self.wandb.log({"test_c_results": test_c_table})
            if 'test_c_error' in wandb_input:
                self.wandb.log({"test/corruption_error: ": wandb_input['test_c_error']})
            if 'test_c_acc' in wandb_input:
                self.wandb.log({"test/corruption_error: ": 100 - 100. * wandb_input['test_c_acc']})
            if 'test_c_cm' in wandb_input:
                test_c_plt = plot_confusion_matrix(wandb_input['test_c_cm'])
                self.wandb.log({'corruption': test_c_plt})

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
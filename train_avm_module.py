from avm_modules import AVMProjModule

import pytorch_lightning as pl 
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

seed_everything(1, workers=True) # seed_everything，讓每次模型訓練都會產生一樣的結果，增進reproducability。

# 建立pytorch lightning模組 
project_module = AVMProjModule(hidden_dim=5, batch_size = 1)

# 建立Tensorboard logger 
logger = TensorBoardLogger('/home/ai/work/logs/tensorboard', 
                           # 儲存tensorboard log的根目錄
                           'multi-task', 
                           # 儲存此專案log的目錄 
                           default_hp_metric=False, 
                           # 若於lightning module中，有手動進行hyperparameter的logging的話，此處設為False 
                           log_graph=True) 
                           # 是否要於tensorboard查看模型架構 
    
# 在訓練過程中，把最好的模型參數存成checkpoint，以幫助事後快速載入
checkpoint = ModelCheckpoint(
    monitor='best_val_mae',
    mode='min',
    dirpath='./checkpoint',
    filename='epoch{epoch:02d}-loss{best_val_mae:.2f}',
    save_last=True,
    auto_insert_metric_name=False,
    every_n_val_epochs=1,
    verbose=True
)

if __name__ == "__main__":
    # 若有使用learning rate scheduler的話，可以用LearningRateMonitor把lr的變化log起來。
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum = True)

    # 加入early_stopping的機制
    early_stopping = EarlyStopping('best_val_mae', mode='min', patience = 5, verbose=True)

    trainer = pl.Trainer(
        # auto_scale_batch_size='power', 
        # 自動選擇最大的batch_size 
        auto_lr_find=True, 
        # 自動選擇最好的learning rate 
        logger = logger, 
        callbacks=[lr_monitor, early_stopping, checkpoint], 
        deterministic=True, 
        # 設定為deterministic，並且搭配前面的seed_everything，讓每次模型訓練都會產生一樣的結果，增進reproducability。
        check_val_every_n_epoch=50,
        progress_bar_refresh_rate=10000
    )

    trainer.tune(project_module)    
    print('Hyper-parameters tuned')

    trainer.fit(project_module)

else:
    project_module = AVMProjModule.load_from_checkpoint('./checkpoint/last.ckpt')
    trainer = pl.Trainer(
        deterministic=True
    )
    trainer.test(project_module)
    

from avm_modules import AVMProjModule
import pytorch_lightning as pl 


project_module = AVMProjModule(hidden_dim=5)

# project_module = AVMProjModule.load_from_checkpoint('lightning_logs/version_34/checkpoints/epoch=12964-step=12964.ckpt')

trainer = pl.Trainer(
    # auto_scale_batch_size='power', 
    auto_lr_find=True, 
    progress_bar_refresh_rate=10000, 
    min_epochs=100000000, 
    max_epochs=10000000000
)


trainer.fit(project_module)

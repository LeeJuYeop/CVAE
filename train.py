from dataset import train_loader
from model import CDCGAN

import lightning as L

num_classes = 10
nz = 5
model = CDCGAN(num_classes=num_classes, nz=nz)

trainer = L.Trainer(max_epochs=3, accelerator="auto")
trainer.fit(model, train_dataloaders=train_loader)
trainer.save_checkpoint("./cdcgan.ckpt")

print(model.device)
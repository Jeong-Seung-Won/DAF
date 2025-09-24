import time, math, copy, importlib, pprint, os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from config import MODEL_ZOO, CFG_REGISTRY     
from utils import (                 
    build_train_loader, build_val_loader, build_test_loader,
    train_epoch, eval_epoch, eval_epoch_denorm
)

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
LR             = 1e-4
EPOCHS         = 50
EARLY_PATIENCE = 5

criterion      = nn.MSELoss()
criterion_mae  = nn.L1Loss()

print("Sanity-checking model instantiation")
for key, cfg in CFG_REGISTRY.items():
    model_cls = MODEL_ZOO[key]
    model     = model_cls(cfg)                
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  {key:<15s}  →  {n_params:,} parameters")   

train_loader = build_train_loader()
val_loader   = build_val_loader()
test_loader  = build_test_loader()

print(f"Train batch es : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")
print(f"Test  batches : {len(test_loader)}\n")

results = {}
best_overall = {"val": math.inf}

for key, base_cfg in CFG_REGISTRY.items():
    print(f"Training model = {key}")
    cfg   = copy.deepcopy(base_cfg)
    model_cls = MODEL_ZOO[key]
    model = model_cls(cfg).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_val = math.inf
    patience = 0

    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader)*EPOCHS)
    scaler = GradScaler(enabled=DEVICE.startswith("cuda"))
    
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_mse, tr_mae = train_epoch(model, train_loader, optimizer, scaler,
                                     criterion, criterion_mae)
        val_mse, val_mae = eval_epoch(model, val_loader,
                                      criterion, criterion_mae)
        scheduler.step()
        
        print(f"[{key}] Ep {epoch:02}/{EPOCHS} | "
              f"Train MSE {tr_mse:.8f} | Val MSE {val_mse:.8f} | "
              f"{(time.time()-t0):.2f}s")
        
        # early-stop
        if val_mse < best_val - 1e-5:
            best_val = val_mse
            patience = 0
            torch.save(model.state_dict(), f"best_{key}_5B.pt")
        else:
            patience += 1
            if patience >= EARLY_PATIENCE:
                print(f"early stop ({key})")
                break
    
    # ─ test best ckpt
    best_model = model_cls(cfg).to(DEVICE)
    state_dict = torch.load(f"best_{key}_5B.pt", map_location=DEVICE)

    
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    best_model.load_state_dict(state_dict)

    test_mse, test_mae = eval_epoch(best_model, test_loader,
                                    criterion, criterion_mae)
    
    print(f"Finished {key:15s} → Test MSE: {test_mse:.8f}  |  MAE: {test_mae:.8f}")
    
    results[key] = {
        "val_mse": best_val,
        "test_mse": test_mse,
        "test_mae": test_mae,
    }
    
    
    if best_val < best_overall["val"]:
        best_overall = {"model": key, "val": best_val,
                        "test_mse": test_mse, "test_mae": test_mae}

print("\n Summary (lowest val MSE):")
for k, v in results.items():
    print(f"{k:15s}  Val MSE {v['val_mse']:.8f} | "
          f"Test MSE {v['test_mse']:.8f}  MAE {v['test_mae']:.8f}")

print("\n Best model:", best_overall)


class MetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}")
        print(f"  Train Loss: {logs.get('loss'):.4f}")
        print(f"  Val Loss: {logs.get('val_loss'):.4f}")
        print(f"  Train Accuracy: {logs.get('accuracy'):.4f}")
        print(f"  Val Accuracy: {logs.get('val_accuracy'):.4f}")
        print(f"  Train Dice Coefficient: {logs.get('dice_coefficient'):.4f}")
        print(f"  Val Dice Coefficient: {logs.get('val_dice_coefficient'):.4f}")
        print(f"  Train Precision: {logs.get('precision'):.4f}")
        print(f"  Val Precision: {logs.get('val_precision'):.4f}")
        print(f"  Train Recall: {logs.get('recall'):.4f}")
        print(f"  Val Recall: {logs.get('val_recall'):.4f}")
        print(f"  Train IoU: {logs.get('iou_metric'):.4f}")
        print(f"  Val IoU: {logs.get('val_iou_metric'):.4f}")
        print(f"  Train Sensitivity: {logs.get('sensitivity'):.4f}")
        print(f"  Val Sensitivity: {logs.get('val_sensitivity'):.4f}")
        print(f"  Train Dice Loss: {logs.get('dice_loss'):.4f}")
        print(f"  Val Dice Loss: {logs.get('val_dice_loss'):.4f}")
        print(f"  Train Binary Loss: {logs.get('binary_loss'):.4f}")
        print(f"  Val Binary Loss: {logs.get('val_binary_loss'):.4f}")
        print("-" * 50)


import torch
from model import AutoregressiveModel
from data import EarthquakeDataset
from test import test_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
import os
import logging

def main():
    # define model parameters
    window_length = 25 # lstm input dim
    hidden_size = 15 # lstm hidden dim
    num_layers = 3 # lstm layer count

    # define training parameters
    batch_size = 16 # batch_size in training
    lr_rate = 0.01 # learning rate
    num_epochs = 200 # num of training epochs
    test_iter = 10000 # test frequency

    # define data parameters
    output_dir = "/Users/furkancoskun/Desktop/doktora_dersler/MMI-711/hw1/training_output"
    hw_data_path = "/Users/furkancoskun/Desktop/doktora_dersler/MMI-711/hw1/hw data"
    train_txts = [  "20170724065925_0919.txt",
                    "20170724065925_0920.txt",
                    "20170724065925_4808.txt",
                    "20170724065925_4809.txt",
                    "20170724065925_4810.txt",
                    "20170724065925_4812.txt",
                    "20170724065925_4814.txt",
                    "20170724065925_4815.txt",
                    "20170724065925_4817.txt",
                    "20170724065925_4819.txt",
                    "20170724065925_4821.txt",
                    "20170724065925_4822.txt" 
                 ]
    test_txts = ["20170724065925_4823.txt"]

    accelerator = Accelerator(
        log_with="wandb",
    )

    ckpt_out_dir = os.path.join(output_dir, "ckpt")
    logger = get_logger(__name__, log_level="INFO")
    if accelerator.is_main_process:
        os.makedirs(ckpt_out_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

    model = AutoregressiveModel(input_dim=window_length,
                                hidden_size=hidden_size,
                                num_layers=num_layers)
    
    train_dataset = EarthquakeDataset(hw_data_path, train_txts, window_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EarthquakeDataset(hw_data_path, test_txts, window_length)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs*0.3),int(num_epochs*0.5),int(num_epochs*0.8)], gamma=0.1)                                        

    train_loader, test_loader, model, optimizer, lr_scheduler = accelerator.prepare(train_loader, test_loader, model, optimizer, lr_scheduler)

    num_training_steps = num_epochs * len(train_loader)
    total_batch_size = batch_size * accelerator.num_processes
    progress_bar = tqdm(range(num_training_steps),disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    logger.info(f"  Test frequency = {test_iter}")

    if accelerator.is_main_process:
        accelerator.init_trackers("earthquake-autoregressiveModel")

    global_step = 0

    for epoch in range(num_epochs):
        for iter, batch in enumerate(train_loader):
            model.train()
            with accelerator.accumulate(model):
                sequence, next_value = batch
                outputs = model(sequence)
                outputs = outputs[:,0,0]
                next_value = next_value
                # loss = torch.nn.functional.mse_loss(outputs, next_value)
                loss = torch.nn.functional.l1_loss(outputs, next_value)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    train_log = {"train_loss": loss.item()}
                    progress_bar.set_postfix(**train_log)
                    accelerator.log(train_log, step=global_step)

                # test model in test_iter frequency
                if accelerator.is_main_process and (global_step % test_iter == 0):
                    accelerator.wait_for_everyone()
                    logger.info(f"Test for {global_step} step")
                    model.eval()
                    l1_dist, l2_dist, _, _ = test_model(model, test_loader)
                    logger.info(f"  l1 distance for {global_step}th step : {l1_dist}")
                    logger.info(f"  l2 distance for {global_step}th step : {l2_dist}")
                    test_log = {"test_l1_dist": l1_dist.item(),
                                "test_l2_dist": l2_dist.item()}
                    accelerator.log(test_log, step=global_step)
                    model.train()

        # save and test after each epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_path = os.path.join(ckpt_out_dir, f"epoch-{epoch}")
            accelerator.save_state(save_path)
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save_model(unwrapped_model, save_path, safe_serialization=False)
            logger.info(f"Saved {epoch} epoch state to {save_path}")
            model.eval()
            l1_dist,l2_dist, _, _ = test_model(model, test_loader)
            logger.info(f"  l1 distance for {epoch}th epoch : {l1_dist}")
            logger.info(f"  l2 distance for {epoch}th epoch : {l2_dist}")
            test_log = {"test_l1_dist_epoch": l1_dist.item(),
                        "test_l2_dist_epoch": l2_dist.item(),}
            accelerator.log(test_log, step=global_step)
            model.train()
            
    accelerator.end_training()

if __name__ == "__main__":
    main()
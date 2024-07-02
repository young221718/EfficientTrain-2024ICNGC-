import torch
import datetime
import os
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        trainloader,
        valloader,
        testloader,
        save_path,
        criterion,
        optimizer,
        scheduler=None,
        max_epoch: int = 300,
        device="cpu",
    ):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.device = device

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_path = f"{save_path}/train.txt"

    def train_one_epoch(self, epoch):
        print(f"Epoch {epoch + 1}/{self.max_epoch}")
        train_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(self.trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.model.train()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        if self.scheduler is not None:
            self.scheduler.step()

        return {"train_loss": train_loss / len(self.trainloader)}

    def validate_one_epoch(self):
        self.model.eval()
        accuracy = 0.0
        running_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.valloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs.data, 1)
                accuracy += (preds == labels).sum().item()

        return {
            "val_loss": running_loss / len(self.valloader),
            "val_accuracy": accuracy / len(self.valloader),
        }

    def train(self):
        val_loss = 100000
        for epoch in range(self.max_epoch):
            train_result = self.train_one_epoch(epoch)
            print(f"epoch{epoch} Train Loss: {train_result['train_loss']:.4f}")
            val_result = self.validate_one_epoch()
            print(
                f"epoch{epoch} Validation Loss: {val_result['val_loss']:.4f} Validation Accuracy: {val_result['val_accuracy']:.4f}"
            )

            if val_loss > val_result["val_loss"]:
                val_loss = val_result["val_loss"]
                torch.save(
                    self.model.state_dict(),
                    f"{self.save_path}/best_model_epoch_{epoch}.pth",
                )

    def save_train_log(self, train_loss, val_loss, val_accuracy):
        # Example usage:
        # save_train_log("train.log", train_result["train_loss"], val_result["val_loss"], val_result["val_accuracy"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = f"{timestamp} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n"
        with open(self.log_path, "a") as file:
            file.write(log)

    def evaluate(self):
        self.model.eval()
        accuracy = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                accuracy += (preds == labels).sum().item()

        print(f"Accuracy: {accuracy / len(self.testloader):.4f}")


class Efficient_Trainer:
    def __init__(
        self,
        model,
        trainloader,
        valloader,
        testloader,
        save_path,
        criterion,
        optimizer,
        scheduler=None,
        max_epoch: int = 300,
        device="cpu",
        eff_list=None,
    ):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.device = device

        self.eff_list = eff_list
        self.train_i = [
            max(len(self.trainloader) - 5 * i, 5) for i in range(len(self.trainloader))
        ]
        print(self.train_i)

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_path = f"{save_path}/train.txt"

    def train_one_epoch(self, epoch):

        print(f"Epoch {epoch}/{self.max_epoch}")
        train_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(self.trainloader):
            indexes, inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.model.train()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            train_loss += loss.item()

            self.eff_list[indexes.tolist()] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.4f}")
                running_loss = 0.0

            if i >= self.train_i[epoch]:
                break

        if self.scheduler is not None:
            self.scheduler.step()

        return {"train_loss": train_loss / len(self.trainloader)}

    def validate_one_epoch(self):
        self.model.eval()
        accuracy = 0.0
        running_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.valloader):
                indexes, inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs.data, 1)
                accuracy += (preds == labels).sum().item()

        return {
            "val_loss": running_loss / len(self.valloader),
            "val_accuracy": accuracy / len(self.valloader),
        }

    def train(self):
        val_loss = 100000
        for epoch in range(self.max_epoch):
            train_result = self.train_one_epoch(epoch)
            print(f"epoch{epoch} Train Loss: {train_result['train_loss']:.4f}")
            val_result = self.validate_one_epoch()
            print(
                f"epoch{epoch} Validation Loss: {val_result['val_loss']:.4f} Validation Accuracy: {val_result['val_accuracy']:.4f}"
            )

            if val_loss > val_result["val_loss"]:
                val_loss = val_result["val_loss"]
                torch.save(
                    self.model.state_dict(),
                    f"{self.save_path}/best_model_epoch_{epoch}.pth",
                )

    def save_train_log(self, train_loss, val_loss, val_accuracy):
        # Example usage:
        # save_train_log("train.log", train_result["train_loss"], val_result["val_loss"], val_result["val_accuracy"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = f"{timestamp} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n"
        with open(self.log_path, "a") as file:
            file.write(log)

    def evaluate(self):
        self.model.eval()
        accuracy = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                accuracy += (preds == labels).sum().item()

        print(f"Accuracy: {accuracy / len(self.testloader):.4f}")

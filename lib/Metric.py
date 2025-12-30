import math
import numpy as np




def ND(prediction_samples, labels):
    median_pred = np.quantile(prediction_samples, 0.5, axis=-1, keepdims = True)
    diff = np.sum(np.abs(median_pred - labels))
    summation = np.sum(np.abs(labels))
    # print("diff", median_pred.shape, labels.shape, (median_pred - labels).shape)
    return diff, summation

def NRMSE(prediction_samples, labels):
    N, T, _ = labels.shape
    median_pred = np.quantile(prediction_samples, 0.5, axis=-1, keepdims = True)
    diff = np.sum((median_pred - labels) ** 2)
    summation = np.sum(np.abs(labels))
    # print(diff.shape)
    return diff, summation, N * T

def CRPS(prediction_samples, labels):
    # ensemble = np.sort(prediction_samples, axis=-1)
    N, T, M = prediction_samples.shape
    N, T, _ = labels.shape
    alpha_array = np.concatenate([np.arange(1/M, 0.999, 1 / M), np.array([0.999])])[np.newaxis, np.newaxis, :]
    pinball_loss = 2 * (alpha_array - (prediction_samples >= labels).astype(float)) * (labels - prediction_samples)
    return np.sum(pinball_loss), (N * T * M)

def QLOSS(prediction_samples, labels, q=0.5):
    quantile = np.quantile(prediction_samples, q, axis=-1, keepdims = True)
    N, T, _ = labels.shape
    qloss = (labels - quantile) * (q - (labels <= quantile).astype(float))
    return np.sum(qloss), N * T

def ECRPS(prediction_samples, labels):
    true = labels
    N, T, M = prediction_samples.shape
    qi = np.expand_dims(prediction_samples, axis=-1)
    qj = np.expand_dims(prediction_samples, axis=-2)
    upper = np.transpose(np.sum(np.abs(prediction_samples - true), axis=-1) / M - 
                        np.sum(np.sum(np.abs(qi - qj), axis=-2), axis=-1) / (2 * M ** 2), (-2, -1))
    
    return np.sum(upper), N * T

def QCL(prediction_samples, labels):
    N, T, M = prediction_samples.shape
    lower = prediction_samples[:, :, :-1]
    upper = prediction_samples[:, :, 1:]
    diff = lower - upper
    zeros = np.zeros_like(diff)
    relu_diff = np.maximum(zeros, diff)
    loss = np.sum(relu_diff)
    return loss, N * T

def MSE(prediction_samples, labels):
    N, T,_ = labels.shape
    median_pred = np.quantile(prediction_samples, 0.5, axis=-1, keepdims = True)
    return np.sum((median_pred - labels) ** 2), N * T

def Rsquare(prediction_samples, labels): # XXXX not appropriate
    N, T,_ = labels.shape
    median_pred = np.quantile(prediction_samples, 0.5, axis=-1, keepdims = True)
    return np.sum((median_pred - labels) ** 2) / np.sum((np.mean(labels) - labels)**2), 1

def MAE(prediction_samples, labels):
    N, T,_ = labels.shape
    median_pred = np.quantile(prediction_samples, 0.5, axis=-1, keepdims = True)
    return np.sum(np.abs(median_pred - labels)), N * T

def MAPE(prediction_samples, labels):
    N, T,_ = labels.shape
    median_pred = np.quantile(prediction_samples, 0.5, axis=-1, keepdims = True)
    return np.sum(np.abs((median_pred - labels) / labels)), N * T

def MSPE(prediction_samples, labels):
    N, T,_ = labels.shape
    median_pred = np.quantile(prediction_samples, 0.5, axis=-1, keepdims = True)
    return np.sum(((median_pred - labels) / labels) ** 2), N * T
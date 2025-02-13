from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Add, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import numpy as np
import os
from sklearn.model_selection import KFold
from multi_fidelity_nn import *

class MultiFidelityLSTM(MultiFidelityNN):
    def __init__(self,
                 input_shapes,
                 coeff,
                 layers_config,
                 train_config,
                 output_units,
                 output_activation,
                 merge_mode='add',
                 correction=False,
                 residual=False,
                 submodel=None,
                 rate=0.2):
        
        super().__init__(input_shapes=input_shapes,
                         coeff=coeff,
                         layers_config=layers_config,
                         train_config=train_config,
                         output_units=output_units,
                         output_activation=output_activation,
                         merge_mode=merge_mode,
                         correction=correction,
                         residual=residual,
                         submodel=submodel,
                         rate=rate)
    
    def build_model(self):
        # Input layers for each fidelity level
        inputs = [Input(shape=shape) for shape in self.input_shapes]
        fidelities =  [item for item in self.layers_config['input_layers']]

        # Hidden layers for each fidelity level
        fidelity_layers = [self._build_hidden_layers(input_layer,fidelity,"input_layers") for (input_layer,fidelity) in zip(inputs,fidelities)]
        
        # Merge fidelity layers
        if self.merge_mode == 'add':
            merged_output = Add()(fidelity_layers)
        elif self.merge_mode == 'concat':
            merged_output = Concatenate()(fidelity_layers)
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}. Use 'add' or 'concat'.")
        
        merged_output = self._build_hidden_layers(merged_output, "output", "output_layers")
        
        if self.residual:
            merged_output = Concatenate()([merged_output] + inputs)
        
        output = Dense(self.output_units, activation=self.output_activation,
                       kernel_regularizer=l2(self.coeff))(merged_output)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=self.train_config.get('optimizer', 'adam'),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        
        print(model.summary())
        return model
    
    def _build_hidden_layers(self, input_layer, fidelity, layers):
        """
        Build hidden layers for each input fidelity level.

        :param input_layer: Input layer for a fidelity level.
        :return: Final hidden layer for the corresponding fidelity level.
        """
        x=input_layer
        print(layers)
        print(fidelity)
        for layer in self.layers_config[layers][fidelity]:
            print(layer)
            if layer["type"] == "Dense":
                x = Dense(layer['units'], 
                        activation=layer['activation'], 
                        kernel_regularizer=l2(self.coeff), kernel_initializer='glorot_uniform')(x)
            elif layer["type"] == "LSTM":
                x = LSTM(layer['units'], return_sequences=True,
                        activation=layer['activation'],
                        kernel_regularizer=l2(self.coeff))(x)
            x = Dropout(rate=layer['rate'])(x)
        return x

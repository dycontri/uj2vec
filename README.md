# uj2vec

Application of Transformer neural networks for representation learning on user journeys (product telemetry event sequences). Makes use of TransformerXL for learning representations that span multiple segments to accomodate very long user journeys. 

## Table of Contents

- Prerequisites
- Installation
- Usage
- Contributing
- License

## Prerequisites

List of things that the user should have installed before they start using your project.

## Installation

Instructions on how to install your project and get it running.

## Usage

Encode a batch of event sequence tokens with shape according to length of the sequence and batch size
```
event_tokens = torch.rand(100, 10).int() # [S, B]
z = model.encoder(event_tokens) # [1, B, Z]
```

Project into joint space between current and random previous view
```
u = model.encoder.project_c_to_joint_space(z_i)
v = model.encoder.project_p_to_joint_space(z_i_minus_n)
```

Reset encoder memory
```
model.encoder.mems = None
```

## Contributing

Instructions on how others can contribute to your project.

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details.

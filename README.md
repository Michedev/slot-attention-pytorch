# Slot attention implementation


## How to install

1) Install through pip


       pip install slot-attention-pytorch



2) Clone the repository
   

       git clone git@github.com:Michedev/slot-attention-pytorch.git

## How to use

You can instantiate Slot Attention as described in the paper with the following lines

    from slot_attention.model import SlotAttentionAE

    model = SlotAttentionAE.from_config()  # default model 

    model = SlotAttentionAE.from_config('clevr_6')  # model with clevr 6 

    
Or alternatively, you can use the paper configuration with a custom dataset

    from slot_attention.model import SlotAttentionAE

    model = SlotAttentionAE.from_config(dataset_width=32, dataset_height=32, max_num_objects=10)  # model with custom-sized dataset

   
If you want to change the model layout, there are two ways:

1) Take and modify the model config under _slot_attention/config/slot_attention.yaml_ and then

       from slot_attention.model import SlotAttentionAE

       custom_config = """
       dataset:
         width: 32
         height: 32
         max_num_objects: 3
       model:
         _target_: slot_attention.model.SlotAttentionAE
         encoder:
           _target_: torch.nn.Sequential
           _args_:
             - _target_: torch.nn.Conv2d
               in_channels: ${model.input_channels}
               out_channels: 16
               kernel_size: 5
               padding: 2
             - _target_: torch.nn.ReLU
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 16
               kernel_size: 5
               padding: 2
             - _target_: torch.nn.ReLU
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 16
               kernel_size: 5
               padding: 2
             - _target_: torch.nn.ReLU
             - _target_: slot_attention.positional_embedding.PositionalEmbedding
               width: ${dataset.width}
               height: ${dataset.height}
               channels: 16
             - _target_: torch.nn.Flatten
               start_dim: 2
               end_dim: 3
             - _target_: torch.nn.GroupNorm
               num_groups: 1
               num_channels: 16
               affine: true
               eps: 0.001
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 16
               kernel_size: 1
             - _target_: torch.nn.ReLU
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 16
               kernel_size: 1
         decoder:
           _target_: torch.nn.Sequential
           _args_:
             - _target_: slot_attention.positional_embedding.PositionalEmbedding
               width: ${dataset.width}
               height: ${dataset.height}
               channels: 16
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 16
               kernel_size: 5
               padding: 2
             - _target_: torch.nn.ReLU
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 16
               kernel_size: 5
               padding: 2
             - _target_: torch.nn.ReLU
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 16
               kernel_size: 5
               padding: 2
             - _target_: torch.nn.ReLU
             - _target_: torch.nn.Conv2d
               in_channels: 16
               out_channels: 4
               kernel_size: 3
               padding: 1
         slot_attention_module:
           _target_: slot_attention.slot_attention_module.SlotAttentionModule
           num_slots: ${dataset.max_num_objects}
           channels_enc: 16
           latent_size: 16
           attention_iters: 5
           eps: 1e-8
           mlp_size: 128
         w_broadcast: 10
         h_broadcast: 10
         width: 16
         height: 16
         input_channels: 3
       """
       custom_config: DictConfig = OmegaConf.create(custom_config)
       model = SlotAttentionAE.from_custom_config(custom_config)


or alternatively, you can instantiate through the constructor

    from slot_attention.model import SlotAttentionAE


    model = SlotAttentionAE(width, height, encoder, decoder, slot_attention_module)

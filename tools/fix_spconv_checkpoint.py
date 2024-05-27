import argparse
import torch


reshape_layers = [
    'unet.blocks.block0.conv_branch.2.weight',
    'unet.blocks.block0.conv_branch.5.weight',
    'unet.blocks.block1.conv_branch.2.weight',
    'unet.blocks.block1.conv_branch.5.weight', 'unet.conv.2.weight',
    'unet.u.blocks.block0.conv_branch.2.weight',
    'unet.u.blocks.block0.conv_branch.5.weight',
    'unet.u.blocks.block1.conv_branch.2.weight',
    'unet.u.blocks.block1.conv_branch.5.weight', 'unet.u.conv.2.weight',
    'unet.u.u.blocks.block0.conv_branch.2.weight',
    'unet.u.u.blocks.block0.conv_branch.5.weight',
    'unet.u.u.blocks.block1.conv_branch.2.weight',
    'unet.u.u.blocks.block1.conv_branch.5.weight', 'unet.u.u.conv.2.weight',
    'unet.u.u.u.blocks.block0.conv_branch.2.weight',
    'unet.u.u.u.blocks.block0.conv_branch.5.weight',
    'unet.u.u.u.blocks.block1.conv_branch.2.weight',
    'unet.u.u.u.blocks.block1.conv_branch.5.weight',
    'unet.u.u.u.conv.2.weight',
    'unet.u.u.u.u.blocks.block0.conv_branch.2.weight',
    'unet.u.u.u.u.blocks.block0.conv_branch.5.weight',
    'unet.u.u.u.u.blocks.block1.conv_branch.2.weight',
    'unet.u.u.u.u.blocks.block1.conv_branch.5.weight',
    'unet.u.u.u.deconv.2.weight',
    'unet.u.u.u.blocks_tail.block0.i_branch.0.weight',
    'unet.u.u.u.blocks_tail.block0.conv_branch.2.weight',
    'unet.u.u.u.blocks_tail.block0.conv_branch.5.weight',
    'unet.u.u.u.blocks_tail.block1.conv_branch.2.weight',
    'unet.u.u.u.blocks_tail.block1.conv_branch.5.weight',
    'unet.u.u.deconv.2.weight',
    'unet.u.u.blocks_tail.block0.i_branch.0.weight',
    'unet.u.u.blocks_tail.block0.conv_branch.2.weight',
    'unet.u.u.blocks_tail.block0.conv_branch.5.weight',
    'unet.u.u.blocks_tail.block1.conv_branch.2.weight',
    'unet.u.u.blocks_tail.block1.conv_branch.5.weight',
    'unet.u.deconv.2.weight', 'unet.u.blocks_tail.block0.i_branch.0.weight',
    'unet.u.blocks_tail.block0.conv_branch.2.weight',
    'unet.u.blocks_tail.block0.conv_branch.5.weight',
    'unet.u.blocks_tail.block1.conv_branch.2.weight',
    'unet.u.blocks_tail.block1.conv_branch.5.weight', 'unet.deconv.2.weight',
    'unet.blocks_tail.block0.i_branch.0.weight',
    'unet.blocks_tail.block0.conv_branch.2.weight',
    'unet.blocks_tail.block0.conv_branch.5.weight',
    'unet.blocks_tail.block1.conv_branch.2.weight',
    'unet.blocks_tail.block1.conv_branch.5.weight', 'input_conv.0.weight'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.in_path)
    key = 'state_dict'  # 'model' for SSTNet
    for layer in reshape_layers:
        checkpoint[key][layer] = checkpoint[key][layer].permute(1, 2, 3, 4, 0)
    torch.save(checkpoint, args.out_path)

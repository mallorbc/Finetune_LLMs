import torch
import numpy as np
import jax.numpy as jnp
import io
import os

torch.set_printoptions(linewidth=130, sci_mode=False)
np.set_printoptions(linewidth=130, suppress=True)

layers = 28
total_shards = 8
ckpt_dir = "step_383500/"
output_dir = "j6b_ckpt"

def reshard(x, old_shape):
    if len(x.shape) == 1:
        # print("epoch")
        # print(x)
        out = x[0:1]

    elif len(x.shape) == 2:
        #print(f"LN/bias {x.shape}")
        #print(x[:, :16])

        if (x[1:] == x[-1]).all():
            #print("LN")
            if (x[1:] == 0).all() or (x[1:] == 1).all():
                out = x[0:1]
            else:
                #print("shard bias")
                out = x[0:1] * 8#* x.shape[0] / old_shape[0]
        else:
            #print("bias")
            out = x.reshape(old_shape)

        #print(out[:, :16])

    elif len(x.shape) == 3:
        #print(f"weight {x.shape}")
        if x.shape[0] * x.shape[2] == old_shape[2]:
            #print("case 1")
            out = jnp.transpose(x, (1, 0, 2)).reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            #print("case 2")
            out = x.reshape(old_shape)
        else:
            raise Exception(f"unimplemented, {x.shape}, {old_shape}")
    else:
        raise Exception(f"unimplemented, {x}")
    #flattened, structure = jax.tree_flatten(out)
    #return flattened
    return out

def get_old_shape(t, dim=2):
    if len(t.shape) == 3:
        shard_shape = t.shape
        if dim == 1:
            return (shard_shape[0] * shard_shape[1], shard_shape[2])
        elif dim == 2:
            return (shard_shape[1], shard_shape[0] * shard_shape[2])
        else:
            raise ValueError(f"unsupported dim {dim}")
    if len(t.shape) == 2:
        return (t.shape[1] * t.shape[0],)
    else:
        raise ValueError(f"unsupported shape {t.shape}")

def read_shard(ckpt_dir):
    global part
    out = []
    idx = part
    file_path = ckpt_dir + f"{idx}.npz"
    #print(f"-- {file_path}")
    with open(file_path, "rb") as f:
        buf = f.read()
        f_io = io.BytesIO(buf)
        deserialized = np.load(f_io)
        for i in deserialized:
            out.append(deserialized[i])
            #print(deserialized[i].shape)
    return out

def save(ckpt):
    try: os.mkdir(output_dir)
    except: pass
    checkpoint = {}
    for i, x in enumerate(ckpt.items()):
        checkpoint[x[0]] = f"{output_dir}/b{i}.pt"
        torch.save(x[1], f"{output_dir}/b{i}.pt")
    torch.save(checkpoint, f"{output_dir}/m.pt")

unshard = None
transforms = [("transformer.wte.bias", None, None), ("transformer.wte.weight", unshard, 1)]

checkpoint = {}

layer_names = sorted(map(str, range(layers)))
for layer in layer_names:
    checkpoint[f"transformer.h.{layer}.attn.attention.bias"] = torch.tril(torch.ones(1, 1, 2048, 2048))
    checkpoint[f"transformer.h.{layer}.attn.attention.masked_bias"] = torch.tensor(-1e9)
    transforms.extend([
        (f"transformer.h.{layer}.attn.attention.q_proj.weight", unshard, 2),
        (f"transformer.h.{layer}.attn.attention.v_proj.weight", unshard, 2),
        (f"transformer.h.{layer}.attn.attention.k_proj.weight", unshard, 2),
        (f"transformer.h.{layer}.attn.attention.out_proj.weight", unshard, 1),
        (f"transformer.h.{layer}.mlp.c_fc.bias", unshard, 1),
        (f"transformer.h.{layer}.mlp.c_fc.weight", unshard, 2),
        (f"transformer.h.{layer}.mlp.c_proj.bias", None, None),
        (f"transformer.h.{layer}.mlp.c_proj.weight", unshard, 1),
        (f"transformer.h.{layer}.ln_1.bias", None, None),
        (f"transformer.h.{layer}.ln_1.weight", None, None),
    ])
transforms.extend([
    ("lm_head.bias", unshard, 1),
    ("lm_head.weight", unshard, 2),
    ("transformer.ln_f.bias", None, None),
    ("transformer.ln_f.weight", None, None),
])

part = 0
element = 0
while len(transforms) > 0:
    print(f"loading shards for part {part}")
    shards = list(map(read_shard, [f"{ckpt_dir}shard_{i}/" for i in range(total_shards)]))
    print(f"read from checkpoint")

    unsharded = []

    for all_shards in zip(*shards):
        x = np.stack(all_shards)
        # No idea why this is V2...?
        if x.dtype == np.dtype('V2'):
            x.dtype = jnp.bfloat16
        x = x.astype(np.float32)
        unsharded.append(x)
        #print(f"unsharded: {x.shape}")

    while len(transforms) > 0 and len(unsharded) > 0:
        transform = transforms.pop(0)
        params = unsharded.pop(0)
        if transform[2] is not None:
            old_shape = (1,) + get_old_shape(params, transform[2])
        else:
            old_shape = (params.shape[1],)
        print(f"< {params.shape} to {old_shape}")
        params = reshard(params, old_shape).squeeze(0).T
        params = torch.tensor(params.copy()).half()
        if params.isnan().any() or params.isinf().any():
            raise ValueError(f"fp16 over/underflow at {part} {element}")
        checkpoint[transform[0]] = params
        print(f"> {transform[0]} {params.shape}")
        element += 1
    part += 1

checkpoint['transformer.wte.weight'] = (checkpoint['transformer.wte.weight'].T + checkpoint['transformer.wte.bias'])
del checkpoint['transformer.wte.bias']

print(f"left over: {unsharded}")
print("saving")
torch.save(checkpoint, "./gpt-j-6B/pytorch_model.bin") # load as in: https://github.com/finetuneanon/misc/blob/main/SizeTest.ipynb
print("done")
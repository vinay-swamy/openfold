# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests to compare components of OpenFold run with the DeepSpeed memory-efficient
attention kernel, DS4Sci_EvoformerAttention vs. a stock PyTorch attention
implementation.
"""

import unittest
import numpy as np
import pickle
import torch
from torch.nn import functional as F

from openfold.data import data_transforms
from openfold.model.primitives import (
    lecun_normal_init_,
    Attention
)
from openfold.utils.tensor_utils import tensor_tree_map

from tests.config import consts
import tests.compare_utils as compare_utils
from tests.data_utils import random_template_feats, random_attention_inputs


@compare_utils.skip_unless_ds4s_installed()
class TestTritonEvoKernel(unittest.TestCase):
    def compare_attention_forward(self, use_deepspeed_kernel=False, use_triton_kernel=False):
        """Compare attention using different attention kernels."""
        batch_size = consts.batch_size
        n_seq = 18
        n_res = 20
        c_hidden = 32
        no_heads = 4
        eps = 2e-2

        q, kv, mask, biases = random_attention_inputs(
            batch_size=batch_size,
            n_seq=n_seq,
            n=n_res,
            no_heads=no_heads,
            c_hidden=c_hidden,
        )

        a = Attention(
            c_hidden,
            c_hidden,
            c_hidden,
            c_hidden,
            no_heads,
        ).cuda()

        # Change output params init for testing since they are initialized with 'final'
        # init (zeros) Otherwise both will just return zero.

        with torch.no_grad():
            lecun_normal_init_(a.linear_g.weight)
            lecun_normal_init_(a.linear_o.weight)

            real_out = a(q, kv, biases=biases).cpu()

            triton_out = a(q, kv, biases=biases, use_triton_evo_attention=True).cpu()
            ds_out = a(q, kv, biases=biases, use_deepspeed_evo_attention=True).cpu()
        if use_deepspeed_kernel and use_triton_kernel:
            err = torch.max(torch.abs(triton_out - ds_out))
        elif use_deepspeed_kernel:
            err = torch.max(torch.abs(ds_out - real_out))
        elif use_triton_kernel:
            err = torch.max(torch.abs(triton_out - real_out))
        
        print(f"use_deepspeed_kernel={use_deepspeed_kernel}, use_triton_kernel={use_triton_kernel} . fwd err: {err}")
        self.assertTrue(err < eps, f"Error: {err}")

    def test_triton_kernel_vs_vanilla_attention_forward(self):
        """Compare forward pass for regular attention vs. Triton Evoformer kernel."""
        self.compare_attention_forward(use_triton_kernel=True)
    def test_deepseed_kernel_vs_vanilla_attention_forward(self):
        """Compare forward pass for regular attention vs. DeepSpeed Evoformer kernel."""
        self.compare_attention_forward(use_deepspeed_kernel=True)
    def test_triton_kernel_vs_deepseed_attention_forward(self):
        """Compare forward pass for Triton Evoformer kernel vs. DeepSpeed Evoformer kernel."""
        self.compare_attention_forward(use_triton_kernel=True, use_deepspeed_kernel=True)

    def compare_attention_backward(self, gt_fwd_args={}, repro_fwd_args={}):
        """
        Compare backward pass for different attention kernels.
        """
        batch_size = consts.batch_size
        n_seq = 18
        n_res = 20
        c_hidden = 32
        no_heads = 4
        eps = consts.eps

        q, kv, mask, biases = random_attention_inputs(
            batch_size=batch_size,
            n_seq=n_seq,
            n=n_res,
            no_heads=no_heads,
            c_hidden=c_hidden,
            requires_grad=True,
        )

        attn = Attention(
            c_hidden,
            c_hidden,
            c_hidden,
            c_hidden,
            no_heads,
        ).cuda()

        with torch.no_grad():
            lecun_normal_init_(attn.linear_g.weight)
            lecun_normal_init_(attn.linear_o.weight)

        def clone(t):
            # Create new params, clone values
            t = t.clone()
            if t.requires_grad:
                t.retain_grad()
            return t

        def init_attn():
            # Create new attention object with same initial weights
            a_clone = Attention(
                c_hidden,
                c_hidden,
                c_hidden,
                c_hidden,
                no_heads,
            ).cuda()

            a_clone.load_state_dict(attn.state_dict())
            return a_clone

        # Clone param values and run attention with DS kernel
        q_repro = clone(q)
        kv_repro = clone(kv)
        biases_repro = [clone(b) for b in biases]

        a_repro = init_attn()
        out_repro = a_repro(
            q_repro, kv_repro, biases=biases_repro, **repro_fwd_args
        )
        loss_repro = torch.mean(out_repro)
        loss_repro.backward()

        q_gt = clone(q)
        kv_gt = clone(kv)
        biases_gt = [clone(b) for b in biases]

        # Clone param values and run attention without DS kernel
        a_gt = init_attn()
        out_gt = a_gt(q_gt, kv_gt, biases=biases_gt, **gt_fwd_args)

        loss_gt = torch.mean(out_gt)
        loss_gt.backward()

        # Compare the grads of attention inputs
        pairs = zip([q_repro, kv_repro, biases_repro[1]], [q_gt, kv_gt, biases_gt[1]])
        for i, item in enumerate(pairs):
            t_repro, t_gt = item
            err = torch.max(torch.abs(t_repro.grad.cpu() - t_gt.grad.cpu()))
            self.assertTrue(err < eps, f"Error item #{i}: {err}")

        # Compare the grads of model weights
        a_repro_params = dict(a_repro.named_parameters())
        a_gt_params = dict(a_gt.named_parameters())
        for name in a_gt_params:
            t_repro = a_repro_params[name]
            t_gt = a_gt_params[name]
            err = torch.max(torch.abs(t_repro.grad.cpu() - t_gt.grad.cpu()))
            self.assertTrue(err < eps, f"Error item {name}: {err}")
    
    def test_triton_kernel_vs_vanilla_attention_backward(self):
        """Compare backward pass for regular attention vs. triton Evoformer kernel."""
        self.compare_attention_backward(repro_fwd_args={"use_triton_evo_attention": True})
    def test_deepseed_kernel_vs_vanilla_attention_backward(self):
        """Compare backward pass for regular attention vs. DeepSpeed Evoformer kernel."""
        self.compare_attention_backward(repro_fwd_args={"use_deepspeed_evo_attention": True})
    def test_triton_kernel_vs_deepseed_attention_backward(self):
        """Compare backward pass for DeepSpeed vs. triton Evoformer kernel."""
        self.compare_attention_backward(repro_fwd_args={"use_triton_evo_attention": True}, gt_fwd_args={"use_deepspeed_evo_attention": True})
    
    
    


if __name__ == "__main__":
    unittest.main()

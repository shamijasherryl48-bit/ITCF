"""
Unit Tests for ITCF PyTorch Models
===================================
Tests forward passes, embedding projections, attention dimensions,
and output shapes for MLAT, CCST, Fusion, and ITCFModel.
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itcf.models import (
    MolecularLandscapeAwareTransformer,
    CellularCharacteristicsAwareSwinTransformer,
    MultimodalFusion,
    ITCFModel,
    GenomicsOnlyMLAT,
    HistopathologyOnlyCCST,
    BasicTransformer,
    MLATCCSTWithoutContrastive
)


class TestITCFModels(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.embed_dim = 64  # lightweight for fast testing
        self.num_layers = 2
        self.num_heads = 4
        
        self.genomic = torch.randn(self.B, 5)
        self.demo = torch.randn(self.B, 3)
        self.tmb = torch.rand(self.B, 1) * 5.0
        self.images = torch.randn(self.B, 3, 224, 224)

    def test_mlat_forward(self):
        mlat = MolecularLandscapeAwareTransformer(
            num_genes=5,
            num_demographics=3,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            contrastive_dim=32
        )
        O_MCIA, z_g = mlat(self.genomic, self.demo, self.tmb)
        self.assertEqual(O_MCIA.shape, (self.B, self.embed_dim))
        self.assertEqual(z_g.shape, (self.B, 32))
        # Verify L2 normalization of contrastive features
        norm = torch.norm(z_g, dim=-1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-5))

    def test_ccst_forward(self):
        ccst = CellularCharacteristicsAwareSwinTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            window_size=7,
            contrastive_dim=32
        )
        O_SW_CPA, z_img = ccst(self.images)
        self.assertEqual(O_SW_CPA.shape, (self.B, self.embed_dim))
        self.assertEqual(z_img.shape, (self.B, 32))

    def test_fusion_forward(self):
        fusion = MultimodalFusion(embed_dim=self.embed_dim, num_classes=3)
        O_g = torch.randn(self.B, self.embed_dim)
        O_img = torch.randn(self.B, self.embed_dim)
        logits, probs, surr, fused = fusion(O_g, O_img)
        
        self.assertEqual(logits.shape, (self.B, 3))
        self.assertEqual(probs.shape, (self.B, 3))
        self.assertEqual(surr.shape, (self.B, 1))
        # Verify probability sums to 1
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_itcf_unified_forward(self):
        model = ITCFModel(
            num_genes=5,
            num_demographics=3,
            num_classes=3,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            contrastive_dim=32
        )
        out = model(self.genomic, self.demo, self.tmb, self.images)
        self.assertIn('logits', out)
        self.assertIn('probs', out)
        self.assertIn('surrogate_score', out)
        self.assertEqual(out['logits'].shape, (self.B, 3))
        
        # Test contrastive loss calculation
        loss = model.compute_contrastive_loss(out['z_genomic'], out['z_image'])
        self.assertGreater(loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()

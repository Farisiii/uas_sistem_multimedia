import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.fft import dct, idct

class JPEGProcessor:
    STANDARD_QUANTIZATION_TABLE = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float64)
    
    def __init__(self, quality=50):
        self.quality = quality
        self.quantization_table = self._scale_quantization_table()
    
    def _scale_quantization_table(self):
        scale = 5000 / self.quality if self.quality < 50 else 200 - 2 * self.quality
        scaled = np.floor((self.STANDARD_QUANTIZATION_TABLE * scale + 50) / 100)
        scaled[scaled == 0] = 1
        return scaled.astype(np.int32)
    
    def dct_2d(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def idct_2d(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def quantize(self, dct_block):
        return np.round(dct_block / self.quantization_table).astype(np.int32)
    
    def dequantize(self, quantized_block):
        return quantized_block.astype(np.float64) * self.quantization_table.astype(np.float64)
    
    def encode_block(self, block):
        shifted = block.astype(np.float64) - 128
        dct_block = self.dct_2d(shifted)
        quantized = self.quantize(dct_block)
        
        return {
            'original': block.astype(np.int32),
            'shifted': shifted,
            'dct': dct_block,
            'quantized': quantized
        }
    
    def decode_block(self, quantized_block):
        dequantized = self.dequantize(quantized_block)
        idct_block = self.idct_2d(dequantized)
        reconstructed = np.clip(idct_block + 128, 0, 255).astype(np.uint8)
        
        return {
            'dequantized': dequantized,
            'idct': idct_block,
            'reconstructed': reconstructed
        }

class ImageLoader:
    @staticmethod
    def load_grayscale(file_path):
        img = Image.open(file_path)
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img, dtype=np.uint8)
        print(f"Image loaded: {img_array.shape}, range: [{img_array.min()}, {img_array.max()}]")
        return img_array
    
    @staticmethod
    def select_random_block(image, block_size=8):
        h, w = image.shape
        if h < block_size or w < block_size:
            raise ValueError(f"Image too small. Minimum {block_size}x{block_size} required")
        
        max_y = h - block_size
        max_x = w - block_size
        
        start_y = np.random.randint(0, max_y + 1)
        start_x = np.random.randint(0, max_x + 1)
        
        block = image[start_y:start_y + block_size, start_x:start_x + block_size]
        print(f"Block selected from position ({start_y}, {start_x})")
        
        return block, (start_y, start_x)

class Visualizer:
    @staticmethod
    def show_block_and_dct(block, quality=50):
        processor = JPEGProcessor(quality)
        encoded = processor.encode_block(block)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        axes[0].imshow(block, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original Block\n(Image)', fontweight='bold')
        axes[0].axis('off')
        
        sns.heatmap(block.astype(np.int32), annot=True, fmt='d', cmap='gray', 
                   ax=axes[1], cbar=False, square=True, annot_kws={'size': 8})
        axes[1].set_title('Original Block\n(Pixel Values)', fontweight='bold')
        axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        sns.heatmap(encoded['dct'], annot=True, fmt='.1f', cmap='RdBu_r', 
                   ax=axes[2], cbar=False, square=True, annot_kws={'size': 7}, center=0)
        axes[2].set_title(f'DCT Coefficients\n(Quality {quality})', fontweight='bold')
        axes[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        plt.suptitle('Original Block and DCT Transform', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return encoded
    
    @staticmethod
    def show_compression_analysis(block, quality=50):
        processor = JPEGProcessor(quality)
        encoded = processor.encode_block(block)
        decoded = processor.decode_block(encoded['quantized'])
        
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)
        
        # Row 1: Quantization process
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(processor.quantization_table, annot=True, fmt='d', cmap='Blues', 
                   ax=ax1, cbar=False, square=True, annot_kws={'size': 8})
        ax1.set_title(f'Quantization Table\n(Quality {quality})', fontweight='bold')
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(encoded['quantized'], annot=True, fmt='d', cmap='RdBu_r', 
                   ax=ax2, cbar=False, square=True, annot_kws={'size': 8}, center=0)
        ax2.set_title(f'Quantized Coefficients\n(Quality {quality})', fontweight='bold')
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax3 = fig.add_subplot(gs[0, 2])
        sns.heatmap(decoded['dequantized'], annot=True, fmt='.1f', cmap='RdBu_r', 
                   ax=ax3, cbar=False, square=True, annot_kws={'size': 6}, center=0)
        ax3.set_title(f'Dequantized Coefficients\n(Quality {quality})', fontweight='bold')
        ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Row 2: Reconstruction
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(decoded['reconstructed'], cmap='gray', vmin=0, vmax=255)
        ax4.set_title(f'Reconstructed Block\n(Quality {quality})', fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        sns.heatmap(decoded['reconstructed'].astype(np.int32), annot=True, fmt='d', cmap='gray', 
                   ax=ax5, cbar=False, square=True, annot_kws={'size': 8})
        ax5.set_title(f'Reconstructed Values\n(Quality {quality})', fontweight='bold')
        ax5.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax6 = fig.add_subplot(gs[1, 2])
        comparison = np.hstack([block, decoded['reconstructed']])
        ax6.imshow(comparison, cmap='gray', vmin=0, vmax=255)
        ax6.set_title('Comparison\n(Original | Reconstructed)', fontweight='bold')
        ax6.axis('off')
        ax6.axvline(x=7.5, color='red', linewidth=2)
        
        plt.suptitle(f'JPEG Encoding-Decoding Analysis - Quality {quality}', 
                    fontsize=16, fontweight='bold')
        plt.show()
        
        return encoded, decoded

def analyze_jpeg_compression(image_path, quality=50):
    loader = ImageLoader()
    viz = Visualizer()
    
    image = loader.load_grayscale(image_path)
    block, position = loader.select_random_block(image)
    
    print(f"\n{'='*80}")
    print(f"JPEG COMPRESSION ANALYSIS - QUALITY {quality}")
    print(f"{'='*80}")
    
    print("\n1. Displaying original block and DCT transform...")
    encoded_original = viz.show_block_and_dct(block, quality)
    
    print(f"\n2. Displaying compression analysis for Quality {quality}...")
    encoded, decoded = viz.show_compression_analysis(block, quality)
    
    print("\n3. Analysis complete!")
    
    return block, encoded, decoded

if __name__ == "__main__":
    IMAGE_PATH = "gambar.jpg"
    original, encoded, decoded = analyze_jpeg_compression(IMAGE_PATH)
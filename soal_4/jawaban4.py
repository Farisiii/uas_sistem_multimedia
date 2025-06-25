import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class AudioWatermarking:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.seed = 104
        np.random.seed(self.seed)
        print(f"Menggunakan seed: {self.seed} (4 Januari)")
        
    def generate_original_signal(self, frequency=440, duration=3):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        signal = np.sin(2 * np.pi * frequency * t)
        return t, signal
    
    def generate_pn_sequence(self, length):
        np.random.seed(self.seed)
        pn_sequence = 2 * np.random.randint(0, 2, length) - 1
        return pn_sequence.astype(np.float64)
    
    def embed_watermark(self, signal, watermark_bit=1, weight=0.1):
        pn_sequence = self.generate_pn_sequence(len(signal))
        
        if watermark_bit == 0:
            watermark_signal = -pn_sequence
        else:
            watermark_signal = pn_sequence
        
        watermarked_signal = signal + weight * watermark_signal
        return watermarked_signal, watermark_signal
    
    def detect_watermark(self, watermarked_signal, weight=0.1):
        pn_sequence = self.generate_pn_sequence(len(watermarked_signal))
        correlation = np.sum(watermarked_signal * pn_sequence) / len(watermarked_signal)
        threshold = weight * 0.5
        
        if correlation > threshold:
            detected_bit = 1
            confidence = correlation / weight
        elif correlation < -threshold:
            detected_bit = 0
            confidence = abs(correlation) / weight
        else:
            detected_bit = None
            confidence = 0
        
        return detected_bit, correlation, confidence
    
    def analyze_watermarking(self, weight=0.1):
        print("=== SIMULASI AUDIO WATERMARKING DENGAN SPREAD SPECTRUM ===\n")
        
        t, original_signal = self.generate_original_signal()
        print(f"Sinyal asli: Sinusoidal 440 Hz, durasi 3 detik")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Jumlah sample: {len(original_signal)}")
        print(f"Bobot watermarking: {weight}\n")
        
        watermarked_signal, watermark_signal = self.embed_watermark(
            original_signal, watermark_bit=1, weight=weight
        )
        
        detected_bit, correlation, confidence = self.detect_watermark(
            watermarked_signal, weight=weight
        )
        
        print(f"--- Hasil Watermarking ---")
        print(f"Bit yang di-embed: 1")
        print(f"Bit yang terdeteksi: {detected_bit}")
        print(f"Korelasi: {correlation:.6f}")
        print(f"Confidence: {confidence:.4f}")
        
        noise_power = np.mean((watermarked_signal - original_signal) ** 2)
        signal_power = np.mean(original_signal ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        print(f"SNR: {snr_db:.2f} dB\n")
        
        self.plot_results(t, original_signal, watermarked_signal, weight)
        self.save_audio_files(original_signal, watermarked_signal, weight)
        
        return original_signal, watermarked_signal, {
            'weight': weight,
            'detected_bit': detected_bit,
            'correlation': correlation,
            'confidence': confidence,
            'snr_db': snr_db
        }
    
    def plot_results(self, t, original_signal, watermarked_signal, weight):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(t[:1000], original_signal[:1000])
        axes[0].set_title('Sinyal Asli (1000 sample pertama)')
        axes[0].set_xlabel('Waktu (s)')
        axes[0].set_ylabel('Amplitudo')
        axes[0].grid(True)
        
        axes[1].plot(t[:1000], watermarked_signal[:1000], label='Watermarked', alpha=0.8)
        axes[1].plot(t[:1000], original_signal[:1000], label='Original', alpha=0.6)
        axes[1].set_title(f'Perbandingan Sinyal (bobot={weight})')
        axes[1].set_xlabel('Waktu (s)')
        axes[1].set_ylabel('Amplitudo')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def test_detection_robustness(self, original_signal, weight=0.1):
        print("=== TEST ROBUSTNESS DETEKSI ===\n")
        
        bits_to_test = [0, 1]
        
        for bit in bits_to_test:
            print(f"Testing dengan watermark bit: {bit}")
            
            watermarked_signal, _ = self.embed_watermark(original_signal, bit, weight)
            detected_bit, correlation, confidence = self.detect_watermark(watermarked_signal, weight)
            
            print(f"  Detected bit: {detected_bit}")
            print(f"  Correlation: {correlation:.6f}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Status: {'✓ CORRECT' if detected_bit == bit else '✗ INCORRECT'}\n")
    
    def save_audio_files(self, original_signal, watermarked_signal, weight):
        original_normalized = original_signal / np.max(np.abs(original_signal))
        watermarked_normalized = watermarked_signal / np.max(np.abs(watermarked_signal))
        
        wavfile.write('original_signal.wav', self.sample_rate, 
                     (original_normalized * 32767).astype(np.int16))
        
        filename = f'watermarked_signal_weight_{weight}.wav'
        wavfile.write(filename, self.sample_rate, 
                     (watermarked_normalized * 32767).astype(np.int16))
        
        print("=== FILE AUDIO TERSIMPAN ===")
        print("- original_signal.wav")
        print(f"- {filename}")
        print("File dapat diputar menggunakan media player\n")

def main(weight=0.1):
    watermarker = AudioWatermarking()
    
    original_signal, watermarked_signal, result = watermarker.analyze_watermarking(weight)
    
    watermarker.test_detection_robustness(original_signal, weight)
    
    print("=== PENJELASAN HASIL ===")
    print("(a) Grafik menunjukkan sinyal sebelum dan sesudah watermarking")
    print("(b) File audio disimpan untuk perbandingan")
    print("(c) Analisis hasil:")
    print(f"   - Bobot: {result['weight']}")
    print(f"   - SNR: {result['snr_db']:.2f} dB")
    print(f"   - Confidence: {result['confidence']:.4f}")
    print(f"   - Deteksi: {'Berhasil' if result['detected_bit'] == 1 else 'Gagal'}")
if __name__ == "__main__":
    main(weight=0.1)
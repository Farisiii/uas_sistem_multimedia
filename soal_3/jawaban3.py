import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import os

class AudioCompressionAnalyzer:
    def __init__(self, wav_file_path):
        self.wav_file_path = wav_file_path
        self.mp3_file_path = wav_file_path.replace('.wav', '_compressed.mp3')
        self.wav_from_mp3_path = wav_file_path.replace('.wav', '_from_mp3.wav')
        
    def compress_to_mp3(self, bitrate="128k"):
        print(f"Compressing {self.wav_file_path} to MP3...")
        
        audio = AudioSegment.from_wav(self.wav_file_path)
        audio.export(self.mp3_file_path, format="mp3", bitrate=bitrate)
        
        mp3_audio = AudioSegment.from_mp3(self.mp3_file_path)
        mp3_audio.export(self.wav_from_mp3_path, format="wav")
        
        print(f"MP3 file saved: {self.mp3_file_path}")
        print(f"WAV from MP3 saved: {self.wav_from_mp3_path}")
    
    def get_file_info(self):
        original_size = os.path.getsize(self.wav_file_path)
        compressed_size = os.path.getsize(self.mp3_file_path)
        
        return {
            'original_mb': original_size / (1024 * 1024),
            'compressed_mb': compressed_size / (1024 * 1024),
            'compression_ratio': original_size / compressed_size,
            'size_reduction_percent': ((original_size - compressed_size) / original_size) * 100
        }
    
    def load_audio(self, file_path):
        sample_rate, data = wavfile.read(file_path)
        
        if len(data.shape) > 1:
            data = data[:, 0]
            
        return sample_rate, data
    
    def calculate_spectrum(self, data, sample_rate):
        data_normalized = data / np.max(np.abs(data))
        fft_data = fft(data_normalized)
        freqs = fftfreq(len(data_normalized), 1/sample_rate)
        
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_data[:len(fft_data)//2])
        
        return positive_freqs, magnitude
    
    def plot_waveforms(self):
        sr_orig, data_orig = self.load_audio(self.wav_file_path)
        sr_comp, data_comp = self.load_audio(self.wav_from_mp3_path)
        
        time_orig = np.linspace(0, len(data_orig)/sr_orig, len(data_orig))
        time_comp = np.linspace(0, len(data_comp)/sr_comp, len(data_comp))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.fill_between(time_orig, data_orig, alpha=0.7, color='blue')
        ax1.set_title('Original Audio Waveform (WAV)', fontsize=14)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(time_comp, data_comp, alpha=0.7, color='red')
        ax2.set_title('Compressed Audio Waveform (MP3)', fontsize=14)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_spectrum_comparison(self):
        sr_orig, data_orig = self.load_audio(self.wav_file_path)
        sr_comp, data_comp = self.load_audio(self.wav_from_mp3_path)
        
        freqs_orig, mag_orig = self.calculate_spectrum(data_orig, sr_orig)
        freqs_comp, mag_comp = self.calculate_spectrum(data_comp, sr_comp)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        ax1.semilogx(freqs_orig, 20*np.log10(mag_orig + 1e-10), 'b-', alpha=0.7)
        ax1.set_title('Original Audio Frequency Spectrum', fontsize=14)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(20, sr_orig//2)
        
        ax2.semilogx(freqs_comp, 20*np.log10(mag_comp + 1e-10), 'r-', alpha=0.7)
        ax2.set_title('Compressed Audio Frequency Spectrum (MP3)', fontsize=14)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(20, sr_comp//2)
        
        ax3.semilogx(freqs_orig, 20*np.log10(mag_orig + 1e-10), 'b-', alpha=0.7, label='Original')
        ax3.semilogx(freqs_comp, 20*np.log10(mag_comp + 1e-10), 'r-', alpha=0.7, label='Compressed')
        ax3.set_title('Frequency Spectrum Comparison', fontsize=14)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude (dB)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xlim(20, min(sr_orig//2, sr_comp//2))
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self, bitrate="128k"):
        print("=== AUDIO COMPRESSION ANALYSIS ===\n")
        
        self.compress_to_mp3(bitrate)
        
        print("--- FILE SIZE COMPARISON ---")
        file_info = self.get_file_info()
        print(f"Original (WAV): {file_info['original_mb']:.2f} MB")
        print(f"Compressed (MP3): {file_info['compressed_mb']:.2f} MB")
        print(f"Compression Ratio: {file_info['compression_ratio']:.2f}:1")
        print(f"Size Reduction: {file_info['size_reduction_percent']:.1f}%")
        
        print("\n--- FREQUENCY SPECTRUM ---")
        self.plot_spectrum_comparison()
        
        print("\n--- WAVEFORM VISUALIZATION ---")
        self.plot_waveforms()
        
        if os.path.exists(self.wav_from_mp3_path):
            os.remove(self.wav_from_mp3_path)

def main():
    wav_file = "dummy_audio.wav"
    
    if not os.path.exists(wav_file):
        print(f"File {wav_file} not found!")
        print("Please replace 'wav_file' with your WAV file path")
        return
    
    analyzer = AudioCompressionAnalyzer(wav_file)
    analyzer.run_analysis(bitrate="320k")

if __name__ == "__main__":
    main()
/// Struktur dasar yang menyimpan parameter umum untuk semua layer saraf
pub struct BaseLayer {
    pub name: String,
    pub trainable: bool,
    pub clip_min: f32,
    pub clip_max: f32,
    pub learning_rate: f32,
}

impl BaseLayer {
    /// Membuat konfigurasi layer dasar baru
    pub fn new(name: &str, learning_rate: f32, clip_min: f32, clip_max: f32) -> Self {
        Self {
            name: name.to_string(),
            trainable: true,
            learning_rate,
            clip_min,
            clip_max,
        }
    }
}

/// Trait untuk fungsi umum yang bisa diterapkan ke semua layer
pub trait Layer {
    /// Mengambil referensi ke konfigurasi dasar
    fn get_base_config(&self) -> &BaseLayer;
    
    /// Mengambil referensi mutabel ke konfigurasi dasar (berguna jika ingin mengubah LR dinamis)
    fn get_base_config_mut(&mut self) -> &mut BaseLayer;
    
    /// Mengambil semua bobot layer sebagai referensi slice array (untuk proses Save / Export)
    fn get_parameters(&self) -> Vec<(&str, &[f32])>;
    
    /// Mengisikan array memori luar ke dalam memori bobot layer (untuk proses Load)
    fn set_parameter(&mut self, name: &str, data: &[f32]) -> Result<(), String>;
    
    /// Menghitung total jumlah elemen parameter bobot di dalam layer ini
    fn count_params(&self) -> usize;
    
    /// Mengembalikan format shape layer ini
    fn get_output_shape(&self) -> String;

    /// Mencetak ringkasan visual untuk layer ini ke console secara langsung
    fn summary(&self) {
        let base = self.get_base_config();
        let divider = "=".repeat(45);
        println!("{}", divider);
        println!(" Layer Name  : {}", base.name);
        println!(" Output Shape: {}", self.get_output_shape());
        println!(" Trainable   : {}", if base.trainable { "Yes" } else { "No" });
        println!(" Total Params: {}", self.count_params());
        println!("{}", divider);
    }
}

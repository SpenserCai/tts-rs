//! Audio utility throughput. These benchmarks do not measure model synthesis.

use std::{hint::black_box, io::Cursor, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tts_core::Audio;

fn waveform(sample_rate: u32, seconds: u32) -> Audio {
    let samples = (0..sample_rate * seconds)
        .map(|sample| {
            let time = sample as f32 / sample_rate as f32;
            (std::f32::consts::TAU * 440.0 * time).sin() * 0.2
        })
        .collect();
    Audio::new(samples, sample_rate).expect("finite benchmark waveform")
}

fn resampling(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("audio/resample");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for input_rate in [44_100, 48_000] {
        for seconds in [1, 3] {
            let audio = waveform(input_rate, seconds);
            group.throughput(Throughput::Elements(audio.samples.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{input_rate}_to_24000"), format!("{seconds}s")),
                &audio,
                |bencher, audio| {
                    bencher.iter(|| {
                        black_box(
                            audio
                                .resample(black_box(24_000))
                                .expect("valid sample rates"),
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

fn wav_io(criterion: &mut Criterion) {
    let audio = waveform(24_000, 1);
    let mut group = criterion.benchmark_group("audio/wav_float32_24000/1s");
    group.throughput(Throughput::Elements(audio.samples.len() as u64));
    group.bench_function("encode", |bencher| {
        bencher.iter(|| {
            let mut output = Cursor::new(Vec::new());
            audio.write_wav(&mut output).expect("in-memory WAV output");
            black_box(output.into_inner())
        });
    });
    let mut bytes = Cursor::new(Vec::new());
    audio.write_wav(&mut bytes).expect("in-memory WAV output");
    let bytes = bytes.into_inner();
    group.bench_function("decode", |bencher| {
        bencher.iter(|| {
            black_box(Audio::read_wav(Cursor::new(black_box(&bytes))).expect("valid WAV fixture"))
        });
    });
    group.finish();
}

criterion_group!(benches, resampling, wav_io);
criterion_main!(benches);

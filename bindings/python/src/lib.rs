//! Python bindings. Long-running native operations detach from the interpreter.

use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use speechfoundry::{Audio, DeviceConfig, Engine, ModelKind, SynthesisOptions, VoicePrompt};

const MAX_PROMPT_BYTES: usize = 16 * 1024 * 1024;
const MAX_OPTIONS_BYTES: usize = 64 * 1024;

fn check_json_size(value: &str, maximum: usize) -> PyResult<()> {
    if value.len() > maximum {
        return Err(value_error(format!(
            "JSON exceeds the {maximum}-byte limit"
        )));
    }
    Ok(())
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Validated mono PCM audio with its explicit sample rate.
#[pyclass(name = "Audio", module = "speechfoundry", frozen)]
pub struct PyAudio {
    inner: Audio,
}

#[pymethods]
impl PyAudio {
    /// Create audio from finite mono PCM samples and a nonzero sample rate.
    #[new]
    fn new(samples: Vec<f32>, sample_rate: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Audio::new(samples, sample_rate).map_err(value_error)?,
        })
    }

    /// Copy mono PCM samples to a Python list.
    #[getter]
    fn samples(&self) -> Vec<f32> {
        self.inner.samples.clone()
    }

    /// Return samples per second.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// Return the duration in seconds.
    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.inner.duration_seconds()
    }

    /// Resample mono audio without holding the Python interpreter lock.
    fn resample(&self, py: Python<'_>, sample_rate: u32) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: self.inner.resample(sample_rate).map_err(value_error)?,
            })
        })
    }
}

/// A loaded native model. Operations on one instance are serialized.
#[pyclass(name = "Engine", module = "speechfoundry", frozen)]
pub struct PyEngine {
    inner: Mutex<Engine>,
}

#[pymethods]
impl PyEngine {
    /// Load an explicit model directory on cpu, cuda:N, or metal:N.
    #[staticmethod]
    #[pyo3(signature = (model_dir, model, device = "cpu"))]
    fn load(py: Python<'_>, model_dir: &str, model: &str, device: &str) -> PyResult<Self> {
        let model: ModelKind = model.parse().map_err(value_error)?;
        let device: DeviceConfig = device.parse().map_err(value_error)?;
        py.detach(|| {
            Ok(Self {
                inner: Mutex::new(Engine::load(model_dir, model, device).map_err(runtime_error)?),
            })
        })
    }

    /// Extract a versioned model-specific voice conditioning JSON document.
    #[pyo3(signature = (samples, sample_rate, transcript = None))]
    fn extract_voice(
        &self,
        py: Python<'_>,
        samples: Vec<f32>,
        sample_rate: u32,
        transcript: Option<&str>,
    ) -> PyResult<String> {
        py.detach(|| {
            let mut engine = self.inner.lock().map_err(runtime_error)?;
            let prompt = engine
                .extract_voice(&samples, sample_rate, transcript)
                .map_err(runtime_error)?;
            serde_json::to_string(&prompt).map_err(runtime_error)
        })
    }

    /// Synthesize speech from text and matching voice conditioning JSON.
    #[pyo3(signature = (text, voice_prompt_json, options_json = None))]
    fn synthesize(
        &self,
        py: Python<'_>,
        text: &str,
        voice_prompt_json: &str,
        options_json: Option<&str>,
    ) -> PyResult<PyAudio> {
        py.detach(|| {
            check_json_size(voice_prompt_json, MAX_PROMPT_BYTES)?;
            if let Some(options) = options_json {
                check_json_size(options, MAX_OPTIONS_BYTES)?;
            }
            let prompt: VoicePrompt =
                serde_json::from_str(voice_prompt_json).map_err(value_error)?;
            prompt.validate().map_err(value_error)?;
            let options = options_json
                .map(serde_json::from_str::<SynthesisOptions>)
                .transpose()
                .map_err(value_error)?
                .unwrap_or_default();
            let mut engine = self.inner.lock().map_err(runtime_error)?;
            Ok(PyAudio {
                inner: engine
                    .synthesize(text, &prompt, &options)
                    .map_err(runtime_error)?,
            })
        })
    }
}

/// Validate a voice prompt and return canonical compact JSON.
#[pyfunction]
fn validate_voice_prompt(py: Python<'_>, voice_prompt_json: &str) -> PyResult<String> {
    py.detach(|| {
        check_json_size(voice_prompt_json, MAX_PROMPT_BYTES)?;
        let prompt: VoicePrompt = serde_json::from_str(voice_prompt_json).map_err(value_error)?;
        prompt.validate().map_err(value_error)?;
        serde_json::to_string(&prompt).map_err(value_error)
    })
}

/// Initialize the extension module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAudio>()?;
    module.add_class::<PyEngine>()?;
    module.add_function(wrap_pyfunction!(validate_voice_prompt, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("_backend_api", 1_u32)?;
    let mut backends = vec!["cpu"];
    if cfg!(feature = "cuda") {
        backends.push("cuda");
    }
    if cfg!(feature = "metal") {
        backends.push("metal");
    }
    module.add("_backends", backends)?;
    Ok(())
}

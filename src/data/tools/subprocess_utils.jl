"""
subprocess_utils.jl — Subprocess execution utilities for bioinformatics tools.
"""

using Logging

"""
    SubprocessResult

Result of running a subprocess.
"""
struct SubprocessResult
    stdout::String
    stderr::String
    exit_code::Int
    cmd::Cmd
end

function Base.show(io::IO, r::SubprocessResult)
    status = r.exit_code == 0 ? "OK" : "FAILED($(r.exit_code))"
    print(io, "SubprocessResult[$status]: $(r.cmd)")
end

"""
    run_subprocess(cmd::Cmd; capture_stdout=true, capture_stderr=true,
                   input_str=nothing, timeout_sec=3600) -> SubprocessResult

Run a command, capturing stdout/stderr. Raises on non-zero exit if check=true.
"""
function run_subprocess(cmd::Cmd;
    capture_stdout::Bool  = true,
    capture_stderr::Bool  = true,
    input_str::Union{String,Nothing} = nothing,
    timeout_sec::Int = 3600,
    check::Bool = true,
)::SubprocessResult
    @debug "Running command: $cmd"
    stdout_buf = capture_stdout ? IOBuffer() : devnull
    stderr_buf = capture_stderr ? IOBuffer() : devnull

    if input_str !== nothing
        stdin_buf = IOBuffer(input_str)
        proc = run(pipeline(stdin_buf, cmd,
                            stdout=capture_stdout ? stdout_buf : devnull,
                            stderr=capture_stderr ? stderr_buf : devnull);
                   wait=true)
    else
        proc = run(pipeline(cmd,
                            stdout=capture_stdout ? stdout_buf : devnull,
                            stderr=capture_stderr ? stderr_buf : devnull);
                   wait=true)
    end

    stdout_str = capture_stdout ? String(take!(stdout_buf)) : ""
    stderr_str = capture_stderr ? String(take!(stderr_buf)) : ""
    exit_code = proc.exitcode

    if check && exit_code != 0
        @error "Command failed (exit code $exit_code): $cmd\nstderr: $stderr_str"
        error("Subprocess failed: $cmd (exit code $exit_code)")
    end

    return SubprocessResult(stdout_str, stderr_str, exit_code, cmd)
end

"""
    run_subprocess_with_tempfile(cmd_builder::Function, input_str::String,
                                 suffix::String; kwargs...) -> SubprocessResult

Run a command that takes a file as input. The input string is written to a temp file,
and `cmd_builder(tmpfile_path)` is called to construct the command.
"""
function run_subprocess_with_tempfile(cmd_builder::Function,
                                       input_str::String,
                                       suffix::String;
                                       kwargs...)::SubprocessResult
    tmpfile = tempname() * suffix
    try
        write(tmpfile, input_str)
        cmd = cmd_builder(tmpfile)
        return run_subprocess(cmd; kwargs...)
    finally
        isfile(tmpfile) && rm(tmpfile, force=true)
    end
end

"""
    check_binary_exists(binary_path::String) -> Bool

Check whether a binary exists and is executable.
"""
function check_binary_exists(binary_path::String)::Bool
    if isabspath(binary_path)
        return isfile(binary_path) && (Sys.isunix() ? filemode(binary_path) & 0o111 != 0 : true)
    else
        # Search in PATH
        path_dirs = split(get(ENV, "PATH", ""), Sys.isunix() ? ':' : ';')
        for dir in path_dirs
            full = joinpath(dir, binary_path)
            isfile(full) && return true
        end
        return false
    end
end

"""
    mktempdir_cleanup(f::Function) -> Any

Execute `f(tmpdir)` with a guaranteed cleanup of tmpdir afterward.
"""
function mktempdir_cleanup(f::Function)
    tmpdir = mktempdir()
    try
        return f(tmpdir)
    finally
        rm(tmpdir, recursive=true, force=true)
    end
end

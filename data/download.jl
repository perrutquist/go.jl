# Script to download the go archives
# Can download all with:
# download_all(cgs_archives, "outdir_path")

using HTTP

# Links for current cgs game server.  More online here:
#cgs_old_url = "http://cgos.boardspace.net/9x9/archive.html"
#cgs_new_url = "http://www.yss-aya.com/cgos/9x9/archive.html"
cgs_archives = [
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2016_03.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2016_02.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2016_01.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2015_12.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2015_11.tar.bz2"
               ]

function fetch_kgs()
    regex = r"(dl\..*\.tar\.bz2)"
    url = "http://www.u-go.net/gamerecords/"
    fetch_urls(url, regex)
end

"Fetch the urls for all the kgs game records"
function fetch_urls(url, regex)
    # Download page source & parse out the record names
    response = HTTP.get(url)
    html = String(response.body)
    fileurls = Vector{AbstractString}()
    offset = 1
    while true
        result = match(regex, html, offset)
        if result != nothing
            push!(fileurls, string("http://", result.match))
            offset = result.offset + 1
        else
            break
        end
    end
    fileurls
end

"Download file and optionally untar/bz2 and remove archive"
function download_archive(url, outdir; expand=false, remove=false)
    !ispath(outdir) && mkdir(outdir)
    name = ascii(split(url, "/")[end])
    outpath = joinpath(outdir, name)

    archive = HTTP.get(url)
    open(outpath, "w") do f
        write(f, archive.body) # Save the payload to a file
    end  
    if expand
        Sys.isapple() ? run(`tar xfz $(outpath) -C $(outdir)`) : run(`tar xjf $(outpath) -C $(outdir)`)
    end
    if remove
        run(`rm $(outpath)`)
    end
end

function download_all(urls, outdir)
    ts = time()
    for (i,url) in enumerate(urls)
        println("$(i)/$(length(urls)): $(url)")
        download_archive(url, outdir, expand=true, remove=true)
    end
    println("Took $(time() - ts) seconds")
end

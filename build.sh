# yum install wget
echo "start mkdir pandoc folder"
mkdir pandoc
echo "create pandoc folder"
pwd
# wget -qO- https://github.com/jgm/pandoc/releases/download/2.11.1.1/pandoc-2.11.1.1-linux-amd64.tar.gz | \
#    tar xvzf - --strip-components 1 -C ./pandoc
tar -zxvf ./pandoc-2.11.1.1-linux-amd64.tar.gz --strip-components 1 -C ./pandoc
ls
export PATH="./pandoc/bin:$PATH"
echo "Pandoc version:"
pandoc --version
npx --version
npx hexo clean
echo "complete clean"
echo "start generate"
npx hexo generate
echo "complete generate"

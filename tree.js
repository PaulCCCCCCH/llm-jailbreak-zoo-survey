const treeData = {
    name: " ",
    children: [
        {
            name: "Background",
            children: [
                {
                    name: "Ethical Alignment",
                    children: [
                        { name: "Prompt-tuning", info: "Info about Prompt-tuning" },
                        { name: "RLHF", info: "Info about RLHF" }
                    ]
                },
                {
                    name: "Jailbreak Process",
                    children: [
                        { name: "Jailbreak LLMS", info: "Info about Jailbreak LLMS" },
                        { name: "Jailbreak VLMS", info: "Info about Jailbreak VLMS" }
                    ]
                }
            ]
        },
        {
            name: "Threats in LLMs",
            children: [
                {
                    name: "Jailbreaks on LLMs",
                    children: [
                        { name: "Gradient-based Jailbreak", info: "Info about Gradient-based Jailbreak" },
                        { name: "Evolutionary-based Jailbreak", info: "Info about Evolutionary-based Jailbreak" },
                        { name: "Demonstration-based Jailbreak", info: "Info about Evolutionary-based Jailbreak" },
                        { name: "Rule-based Jailbreak", info: "Info about Evolutionary-based Jailbreak" },
                        { name: "Multi Agent-based Jailbreak", info: "Info about Evolutionary-based Jailbreak" }
                    ]
                },
                {
                    name: "Defense on LLMs",
                    children: [
                        { name: "Prompt Detection-based Defense", info: "Info about Prompt Detection-based Defense" },
                        { name: "Prompt Perturbation-based Defense", info: "Info about Prompt Perturbation-based Defense" },
                        { name: "Demonstration-based Defense", info: "Info about Prompt Perturbation-based Defense" },
                        { name: "Generation Intervention-based Defense", info: "Info about Prompt Perturbation-based Defense" },
                        { name: "Response Evaluation-based Defense", info: "Info about Prompt Perturbation-based Defense" },
                        { name: "Model Fine Tuning-based Defense", info: "Info about Prompt Perturbation-based Defense" }
                    ]
                }
            ]
        },
        {
            name: "Threats in VLMs",
            children: [
                {
                    name: "Jailbreaks on VLMs",
                    children: [
                        { name: "Prompt-to-image injection Jailbreak", info: "Info about Gradient-based Jailbreak" },
                        { name: "Prompt-Image Perturbation Injection Jailbreak", info: "Info about Evolutionary-based Jailbreak" },
                        { name: "Proxy Model Transfer Jailbreak", info: "Info about Evolutionary-based Jailbreak" }
                    ]
                },
                {
                    name: "Defense on VLMs",
                    children: [
                        { name: "Model Fine Tuning-based Defense", info: "Info about Prompt Detection-based Defense" },
                        { name: "Response Evaluation-based Defense", info: "Info about Prompt Perturbation-based Defense" },
                        { name: "Prompt Perturbation-based Defense", info: "Info about Prompt Perturbation-based Defense" }
                    ]
                }

            ]
        }
    ]
};

const margin = { top: 20, right: 90, bottom: 30, left: 90 };
const width = 1200 - margin.left - margin.right;
const height = 700 - margin.top - margin.bottom;

const svg = d3.select("#tree-container").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

const duration = 750;
const root = d3.hierarchy(treeData, d => d.children);
root.x0 = height / 2;
root.y0 = 0;
const tree = d3.tree().size([height, width]);

let i = 0;

update(root);

function update(source) {
    const treeData = tree(root);
    const nodes = treeData.descendants();
    const links = treeData.descendants().slice(1);

    nodes.forEach(d => { d.y = d.depth * 220; });

    const node = svg.selectAll('g.node')
        .data(nodes, d => d.id || (d.id = ++i));

    const nodeEnter = node.enter().append('g')
        .attr('class', 'node')
        .attr("transform", d => `translate(${source.y0},${source.x0})`)
        .on('click', (event, d) => {
            if (d.children) {
                d._children = d.children;
                d.children = null;
            } else {
                d.children = d._children;
                d._children = null;
            }
            update(d);
            if (d.data.info) {
                const modal = new bootstrap.Modal(document.getElementById('infoModal'));
                document.getElementById('infoModalLabel').textContent = d.data.name;
                document.querySelector('#infoModal .modal-body').textContent = d.data.info;
                modal.show();
            }
        });;

    nodeEnter.append('circle')
        .attr('r', 1e-6)
        .style("fill", d => d._children ? "#a91d3a" : "#eeeeee");

    nodeEnter.append('text')
        .attr("dy", ".35em")
        .attr("x", d => d.children || d._children ? -13 : 13)
        .attr("text-anchor", d => d.children || d._children ? "end" : "start")
        .text(d => d.data.name)
        .style("font-size", '14px')
        .style("fill-opacity", 1e-6)
        .attr("class", "node-label");

    const nodeUpdate = nodeEnter.merge(node);

    nodeUpdate.transition()
        .duration(duration)
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .on('end', function () {
            nodeUpdate.selectAll('circle, text')
                .on('mouseover', function (event, d) {
                    d3.select(this.parentNode).select('circle').transition().duration(200).attr('r', 12).style('fill', '#a91d3a');
                    d3.select(this.parentNode).select('text').transition().duration(200).style('font-size', '16px');
                })
                .on('mouseout', function (event, d) {
                    d3.select(this.parentNode).select('circle').transition().duration(200).attr('r', 10).style('fill', d => d._children ? "#c73659" : "#eeeeee");
                    d3.select(this.parentNode).select('text').transition().duration(200).style('font-size', '14px');
                })
        });

    nodeUpdate.select('circle')
        .attr('r', 10)
        .style("fill", d => d._children ? "#c73659" : "#eeeeee")
        .attr('cursor', 'pointer');

    nodeUpdate.select('text')
        .style("fill-opacity", 1);

    const nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", d => `translate(${source.y},${source.x})`)
        .remove();

    nodeExit.select('circle')
        .attr('r', 1e-6);

    nodeExit.select('text')
        .style('fill-opacity', 1e-6);

    const link = svg.selectAll('path.link')
        .data(links, d => d.id);

    const linkEnter = link.enter().insert('path', "g")
        .attr("class", "link")
        .attr('d', d => {
            const o = { x: source.x0, y: source.y0 };
            return diagonal(o, o);
        });

    const linkUpdate = linkEnter.merge(link);

    linkUpdate.transition()
        .duration(duration)
        .attr('d', d => diagonal(d, d.parent));

    link.exit().transition()
        .duration(duration)
        .attr('d', d => {
            const o = { x: source.x, y: source.y };
            return diagonal(o, o);
        })
        .remove();

    nodes.forEach(d => {
        d.x0 = d.x;
        d.y0 = d.y;
    });

    function diagonal(s, d) {
        return `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
                ${(s.y + d.y) / 2} ${d.x},
                ${d.y} ${d.x}`;
    }
}

anime({
    targets: '#tree-container',
    opacity: [0, 1],
    translateY: [50, 0],
    easing: 'easeOutExpo',
    duration: 1500
});

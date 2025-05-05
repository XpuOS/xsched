#include <memory>
#include <iostream>
#include <CLI/CLI.hpp>

#include "cli.h"
#include "xsched/utils/common.h"
#include "xsched/protocol/def.h"

using namespace xsched::service;

int main(int argc, char **argv)
{
    auto xcli = std::make_unique<CLI::App>("XCLI: Command-line tool for XSched.");

    std::string addr = "127.0.0.1";
    xcli->add_option("-a,--addr", addr, "XSched server ipv4 address. Default is [127.0.0.1].")
        ->check(CLI::ValidIPV4);

    uint16_t port = XSCHED_SERVER_DEFAULT_PORT;
    xcli->add_option("-p,--port", port, "XSched server port. Default is ["
                     TOSTRING(XSCHED_SERVER_DEFAULT_PORT) "].")->check(CLI::Range(0x1U, 0xFFFFU));

    // subcommand top
    CLI::App *top = xcli->add_subcommand("top", "Show the top information of the XQueues.");
    float frequency = 1.0;
    top->add_option("-f,--frequency", frequency,
                    "Frequency of refreshing the information in FPS. Default is [1.0] FPS.")
       ->check(CLI::Range(0.1f, 30.0f));

    // subcommand list
    CLI::App *list = xcli->add_subcommand("list", "List the information of the XQueues.");

    // subcommand config
    CLI::App *config = xcli->add_subcommand("config", "Change the configuration of the XQueues.");
    XQueueHandle xq_cfg = 0;
    config->add_option("-x,--xqueue", xq_cfg, "Handle of the XQueue")->required();
    XPreemptLevel level = XPreemptLevel::kPreemptLevelUnknown;
    config->add_option("-l,--level", level, "Preempt level of the XQueue. "
                       "1: Blocking, 2: Deactivate, 3:Interrupt.")
          ->check(CLI::IsMember({ 1, 2, 3 }));
    int64_t threshold = -1;
    config->add_option("-t,--threshold", threshold, "Command threshold of the XQueue.")
          ->check(CLI::Range(1, 0x10000));
    int64_t batch_size = -1;
    config->add_option("-b,--batch-size", batch_size, "Command batch size of the XQueue.")
          ->check(CLI::Range(1, 0x10000));

    // subcommand policy
    CLI::App *policy = xcli->add_subcommand("policy",
        "Query or set the scheduler policy of the XSched server.");
    CLI::Option *policy_q_op = policy->add_flag("-q,--query",
        "Query the current policy of the XSched server.");
    CLI::Option *policy_s_op = policy->add_flag("-s,--set",
        "Set the policy of the XSched server.");
    std::string policy_name;
    CLI::Option *policy_n_op = policy->add_option(
        "-n,--name", policy_name, "Name of the policy to set.");

    // subcommand hint
    CLI::App *hint = xcli->add_subcommand("hint", "Give a hint to the XSched server.");
    XQueueHandle hint_xq = 0;
    CLI::Option *hint_xq_op = hint->add_option("-x,--xqueue", hint_xq, "Handle of the XQueue.");
    PID hint_pid = 0;
    CLI::Option *hint_pid_op = hint->add_option("--pid", hint_pid, "PID of the process.");
    Priority prio = 0;
    CLI::Option *hint_prio_op = hint->add_option("-p,--priority", prio, "Priority of the XQueue. "
        "Higher value means higher priority.")->check(CLI::Range(PRIORITY_MIN, PRIORITY_MAX));
    Utilization hint_util = 0;
    CLI::Option *hint_util_op = hint->add_option("-u,--utilization", hint_util,
        "Utilization of the XQueue.")->check(CLI::Range(UTILIZATION_MIN, UTILIZATION_MAX));
    Timeslice hint_ts = 0;
    CLI::Option *hint_ts_op = hint->add_option("-t,--timeslice", hint_ts,
        "Timeslice (in us) of the scheduler.")->check(CLI::Range(TIMESLICE_MIN, TIMESLICE_MAX));

    CLI11_PARSE(*xcli, argc, argv);
    Cli cli(addr, port);

    if (top->parsed()) return cli.Top(1000.0 / frequency);
    if (list->parsed()) return cli.ListXQueues();
    if (config->parsed()) return cli.ConfigXQueue(xq_cfg, level, threshold, batch_size);

    if (policy->parsed()) {
        if (!policy_q_op->empty()) return cli.QueryPolicy();
        if (!policy_s_op->empty()) {
            if (!policy_n_op->empty()) return cli.SetPolicy(policy_name);
            std::cout << "Policy name is required for setting the policy." << std::endl;
        }
        std::cout << policy->help();
        return 1;
    }

    if (hint->parsed()) {
        if (!hint_prio_op->empty()) {
            if (!hint_xq_op->empty()) return cli.SetPriority(hint_xq, prio);
            if (!hint_pid_op->empty()) return cli.SetProcessPriority(hint_pid, prio);
            std::cout << "XQueue handle is required for setting the priority." << std::endl;
        }
        if (!hint_util_op->empty()) {
            if (!hint_xq_op->empty()) return cli.SetUtilization(hint_xq, hint_util);
            if (!hint_pid_op->empty()) return cli.SetProcessUtilization(hint_pid, hint_util);
            std::cout << "XQueue handle is required for setting the utilization." << std::endl;
        }
        if (!hint_ts_op->empty()) return cli.SetTimeslice(hint_ts);
        std::cout << hint->help();
        return 1;
    }

    std::cout << xcli->help();
    return 1;
}

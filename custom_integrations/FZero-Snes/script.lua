data.prev_lapcount = 0
data.prev_speed = 0
data.prev_power = 2048
data.prev_rank = 1


function fzero_reward()
    local reward = 0
    -- negative reward if power goes down
    if data.power < data.prev_power then
        local diff = math.abs(data.power - data.prev_power)
        reward = reward - diff
        data.prev_power = data.power
    end
    -- positive reward if power goes up
    if data.power > data.prev_power then
        local diff = math.abs(data.power - data.prev_power)
        reward = reward + diff
        data.prev_power = data.power
    end
    -- negative reward if speed goes down
    if data.speed < data.prev_speed then
        local diff = math.abs(data.speed - data.prev_speed)
        reward = reward - diff
        data.prev_speed = data.speed
    end
    -- positive reward if speed goes up
    if data.speed > data.prev_speed then
        local diff = math.abs(data.speed - data.prev_speed)
        reward = reward + diff
        data.prev_speed = data.speed
    end
    -- positive reward if finishing 1 lap
    -- extra positive reward if finishing a lap in first place
    if data.lapcount > data.prev_lapcount then
        reward = reward + 100
        data.prev_lapcount = data.lapcount
        if data.rank == 1 then
            reward = reward + 400
        end
    end
    -- positive reward if going up in rank
    if data.rank < data.prev_rank then
        reward = reward + 100
        data.prev_rank = data.rank
    end
    -- negative reward if going down in rank
    if data.rank > data.prev_rank then
        reward = reward - 100
        data.prev_rank = data.rank
    end
    -- negative reward if driving in reverse
    if data.reverse == 5 then
        reward = reward - 10
    end
    return reward

end

function fzero_done()
    -- end state if power is almost empty
    if data.power <= 100 then
        return true
    end
    -- end state if finishing all laps
    if data.lapcount == 5 then
        return true
    end
    -- end state if agent gets in last place
    if data.lost >= 1 then
        return true
    end
    -- end state if agent falls of map
    if data.fall == 1 then
        return true
    end
end